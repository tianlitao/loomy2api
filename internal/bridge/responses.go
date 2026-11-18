// Package bridge OpenAI Responses API（Codex）↔ Chat Completions 翻译。
//
// Codex 只说 /v1/responses，上游只有 /chat/completions。这里把请求翻译过去，
// 再把结果翻译成 Responses 的 response 对象与 SSE 事件流。
package bridge

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/misakano7545/loomy2api-panel/internal/account"
)

// ResponsesToChat 把 Responses 请求翻成 Chat Completions 请求。
//
// 支持 input 为字符串或条目数组（message / function_call / function_call_output），
// 以及 instructions（→ system）、tools（扁平 function 形态）、max_output_tokens。
func ResponsesToChat(req map[string]any) (map[string]any, error) {
	messages := []any{}

	if ins := str(req["instructions"]); ins != "" {
		messages = append(messages, map[string]any{"role": "system", "content": ins})
	}

	switch input := req["input"].(type) {
	case string:
		if input != "" {
			messages = append(messages, map[string]any{"role": "user", "content": input})
		}
	case []any:
		for _, raw := range input {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			switch str(item["type"]) {
			case "message", "":
				role := str(item["role"])
				if role == "" {
					role = "user"
				}
				if role == "developer" {
					role = "system"
				}
				content := ContentToText(item["content"])
				msg := map[string]any{"role": role, "content": content}
				if calls, ok := item["tool_calls"].([]any); ok && len(calls) > 0 {
					msg["tool_calls"] = calls
				}
				messages = append(messages, msg)
			case "function_call":
				args := str(item["arguments"])
				if args == "" {
					args = "{}"
				}
				messages = append(messages, map[string]any{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []any{map[string]any{
						"id":   str(item["call_id"]),
						"type": "function",
						"function": map[string]any{
							"name":      str(item["name"]),
							"arguments": args,
						},
					}},
				})
			case "function_call_output":
				messages = append(messages, map[string]any{
					"role":         "tool",
					"tool_call_id": str(item["call_id"]),
					"content":      ContentToText(item["output"]),
				})
			case "reasoning":
				// Codex 的历史会带 reasoning 项。上游没有 reasoning 入参，但把内容
				// 完全丢掉会让工具回放缺少“为什么调用”的上下文；压回 assistant
				// 正文前缀，且不产生独立消息。
				if text := reasoningItemText(item); text != "" {
					messages = append(messages, map[string]any{
						"role": "assistant", "content": text,
					})
				}
			}
		}
	case nil:
		// 无 input：留空让上层报错
	default:
		messages = append(messages, map[string]any{"role": "user", "content": fmt.Sprint(input)})
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("input 不能为空")
	}

	out := map[string]any{
		"model":    req["model"],
		"messages": messages,
		"stream":   truthy(req["stream"]),
	}
	if v, ok := req["max_output_tokens"]; ok {
		out["max_tokens"] = v
	} else if v, ok := req["max_tokens"]; ok {
		out["max_tokens"] = v
	}
	if v, ok := req["temperature"]; ok {
		out["temperature"] = v
	}
	if v, ok := req["top_p"]; ok {
		out["top_p"] = v
	}
	if tools, ok := req["tools"].([]any); ok && len(tools) > 0 {
		converted := make([]any, 0, len(tools))
		for _, rawTool := range tools {
			t, ok := rawTool.(map[string]any)
			if !ok {
				continue
			}
			// Responses 的工具是扁平形态 {type:"function", name, parameters}；
			// 也兼容已嵌套的 {type:"function", function:{...}}。
			name := str(t["name"])
			params := t["parameters"]
			desc := str(t["description"])
			if fn, ok := t["function"].(map[string]any); ok {
				name = str(fn["name"])
				params = fn["parameters"]
				desc = str(fn["description"])
			}
			if name == "" {
				continue
			}
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			converted = append(converted, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": name, "description": desc, "parameters": params,
				},
			})
		}
		if len(converted) > 0 {
			out["tools"] = converted
		}
	}
	if v, ok := req["tool_choice"]; ok {
		out["tool_choice"] = v
	}
	if v, ok := req["parallel_tool_calls"]; ok {
		out["parallel_tool_calls"] = v
	}
	applyReasoning(out, req)
	return out, nil
}

// reasoningItemText 从 Responses 的 reasoning 项里取可读文本（summary/content）。
func reasoningItemText(item map[string]any) string {
	var parts []string
	if summary, ok := item["summary"].([]any); ok {
		for _, raw := range summary {
			if m, ok := raw.(map[string]any); ok {
				if t := str(m["text"]); t != "" {
					parts = append(parts, t)
				}
			}
		}
	}
	if content, ok := item["content"].([]any); ok {
		for _, raw := range content {
			if m, ok := raw.(map[string]any); ok {
				if t := str(m["text"]); t != "" {
					parts = append(parts, t)
				}
			}
		}
	}
	return strings.Join(parts, "\n")
}

// ContentToText 把 Responses 的 content（字符串或条目数组）压成纯文本。
func ContentToText(content any) string {
	switch c := content.(type) {
	case nil:
		return ""
	case string:
		return c
	case []any:
		var parts []string
		for _, raw := range c {
			switch item := raw.(type) {
			case string:
				parts = append(parts, item)
			case map[string]any:
				switch str(item["type"]) {
				case "input_text", "output_text", "text", "":
					if t := str(item["text"]); t != "" {
						parts = append(parts, t)
					}
				case "input_image":
					parts = append(parts, "[image]")
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return fmt.Sprint(content)
}

// cachedTokens 从 OpenAI usage.prompt_tokens_details.cached_tokens 取缓存命中数。
func cachedTokens(usage map[string]any) any {
	if d, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		return d["cached_tokens"]
	}
	return nil
}

// ChatToResponse 把非流式 Chat Completion 翻成 Responses 的 response 对象。
func ChatToResponse(obj map[string]any, model string) map[string]any {
	choices, _ := obj["choices"].([]any)
	var choice map[string]any
	if len(choices) > 0 {
		choice, _ = choices[0].(map[string]any)
	}
	message, _ := choice["message"].(map[string]any)

	output := []any{}
	if r := reasoningText(message); r != "" {
		output = append(output, map[string]any{
			"type":              "reasoning",
			"id":                "rs_" + account.RandomHex(8),
			"summary":           []any{map[string]any{"type": "summary_text", "text": r}},
			"encrypted_content": "reasoning_" + account.RandomHex(16),
		})
	}
	if t := str(message["content"]); t != "" {
		output = append(output, map[string]any{
			"type":   "message",
			"id":     "msg_" + account.RandomHex(8),
			"role":   "assistant",
			"status": "completed",
			"content": []any{map[string]any{
				"type": "output_text", "text": t, "annotations": []any{},
			}},
		})
	}
	if calls, ok := message["tool_calls"].([]any); ok {
		for _, rawCall := range calls {
			call, ok := rawCall.(map[string]any)
			if !ok {
				continue
			}
			fn, _ := call["function"].(map[string]any)
			args := str(fn["arguments"])
			if args == "" {
				args = "{}"
			}
			output = append(output, map[string]any{
				"type":      "function_call",
				"id":        "fc_" + account.RandomHex(8),
				"call_id":   str(call["id"]),
				"name":      str(fn["name"]),
				"arguments": args,
				"status":    "completed",
			})
		}
	}

	usage, _ := obj["usage"].(map[string]any)
	prompt := intOf(usage["prompt_tokens"])
	completion := intOf(usage["completion_tokens"])
	respUsage := map[string]any{
		"input_tokens":          prompt,
		"output_tokens":         completion,
		"total_tokens":          prompt + completion,
		"input_tokens_details":  map[string]any{"cached_tokens": intOf(cachedTokens(usage))},
		"output_tokens_details": map[string]any{"reasoning_tokens": 0},
	}
	if d, ok := usage["completion_tokens_details"].(map[string]any); ok {
		respUsage["output_tokens_details"].(map[string]any)["reasoning_tokens"] = intOf(d["reasoning_tokens"])
	}
	if v, ok := usage["points_consumed"]; ok {
		respUsage["points_consumed"] = v
	}

	name := model
	if m := str(obj["model"]); m != "" {
		name = m
	}
	status := "completed"
	if str(choice["finish_reason"]) == "length" {
		status = "incomplete"
	}
	return map[string]any{
		"id":                 orDefault(str(obj["id"]), "resp_"+account.RandomHex(10)),
		"object":             "response",
		"created_at":         timeOf(obj["created"]),
		"status":             status,
		"model":              name,
		"output":             output,
		"usage":              respUsage,
		"error":              nil,
		"incomplete_details": nil,
	}
}

// ResponsesStreamer 把 chat 增量翻译成 Responses SSE 事件。
type ResponsesStreamer struct {
	Model      string
	ResponseID string
	seq        int

	reasoningID  string
	reasoningIdx int
	reasoningTxt strings.Builder
	toolCalls    int

	messageID string
	textOpen  bool
	fullText  strings.Builder

	// 每个 item 在 response.output 中的固定下标。Codex 会按 output_index
	// 把增量事件归到对应 item；如果收尾时才重新分配下标，前面的 delta 会
	// 落到错误的条目上。
	outputItems   []any
	reasoningOpen bool

	tools       map[int]*toolAccum
	order       []int
	toolIndex   map[int]int
	toolIDs     map[int]string
	toolCallIDs map[int]string
	emitted     map[int]bool

	usage map[string]any

	// finished 表示上游已经给出本轮收口信号（finish_reason 非空）。
	// 只有它为真时才允许把半截工具参数当完整调用发出去。
	finished bool
}

// Truncated 报告上游是否在收口前断流。
//
// 上游（尤其是长参数工具调用）会不发 finish_reason 就断开 TCP；旧实现把读错误
// 当成正常结束，照样发 response.completed，客户端于是拿到半截 arguments 再报
// “failed to parse function arguments: EOF while parsing an object”。这里把
// 「有没有真正收口」暴露出来，由 handler 决定发 response.failed 还是 completed。
func (s *ResponsesStreamer) Truncated() bool {
	// 没有任何工具调用时，正文/思考是否收口由上游的 finish_reason 决定；
	// 缺 finish_reason 但已自然结束的短回合在各家网关都能见到，不该判失败。
	if s.finished || len(s.tools) == 0 {
		return false
	}
	for _, acc := range s.tools {
		if acc == nil {
			continue
		}
		if acc.Name == "" || !json.Valid([]byte(acc.Args.String())) {
			return true
		}
	}
	return false
}

// messageIndex 用固定下标取 message 的 output_index，避免 reasoning 是否出现
// 导致同一条消息在不同事件里拿到不同下标。
func (s *ResponsesStreamer) messageIndex() int {
	for i, item := range s.outputItems {
		if m, ok := item.(map[string]any); ok && str(m["id"]) == s.messageID {
			return i
		}
	}
	return len(s.outputItems) - 1
}

// NewResponsesStreamer 构造翻译器。
func NewResponsesStreamer(model, responseID string) *ResponsesStreamer {
	if responseID == "" {
		responseID = "resp_" + account.RandomHex(10)
	}
	return &ResponsesStreamer{
		Model: model, ResponseID: responseID,
		usage:       map[string]any{},
		tools:       map[int]*toolAccum{},
		toolIndex:   map[int]int{},
		toolIDs:     map[int]string{},
		toolCallIDs: map[int]string{},
		emitted:     map[int]bool{},
	}
}

// SSEEvent 一个 Responses SSE 事件（event 名 + JSON 数据）。
type SSEEvent struct {
	Name string
	Data map[string]any
}

func (s *ResponsesStreamer) next() int { s.seq++; return s.seq }

// Start 发出 response.created 与 response.in_progress。
func (s *ResponsesStreamer) Start() []SSEEvent {
	base := s.envelope("in_progress", []any{})
	return []SSEEvent{
		{Name: "response.created", Data: map[string]any{
			"type": "response.created", "sequence_number": s.next(), "response": base,
		}},
		{Name: "response.in_progress", Data: map[string]any{
			"type": "response.in_progress", "sequence_number": s.next(), "response": base,
		}},
	}
}

func (s *ResponsesStreamer) envelope(status string, output []any) map[string]any {
	usage := s.usageEnvelope()
	return map[string]any{
		"id": s.ResponseID, "object": "response", "status": status,
		"model": s.Model, "output": output, "usage": usage,
		"created_at": timeNow(), "error": nil, "incomplete_details": nil,
	}
}

// usageEnvelope 构造 Codex 严格反序列化所需的完整 usage。
// codex-api 的 ResponseCompletedUsage 要求 input_tokens / output_tokens /
// total_tokens 三个字段都存在；缺 total_tokens 会让整轮在工具执行后被判失败。
func (s *ResponsesStreamer) usageEnvelope() map[string]any {
	prompt := intOf(s.usage["prompt_tokens"])
	completion := intOf(s.usage["completion_tokens"])
	reasoning := 0
	if d, ok := s.usage["completion_tokens_details"].(map[string]any); ok {
		reasoning = intOf(d["reasoning_tokens"])
	}
	usage := map[string]any{
		"input_tokens":          prompt,
		"output_tokens":         completion,
		"total_tokens":          prompt + completion,
		"input_tokens_details":  map[string]any{"cached_tokens": intOf(cachedTokens(s.usage))},
		"output_tokens_details": map[string]any{"reasoning_tokens": reasoning},
	}
	if v, ok := s.usage["points_consumed"]; ok {
		usage["points_consumed"] = v
	}
	return usage
}

// Feed 吃一个 chat 增量。
func (s *ResponsesStreamer) Feed(obj map[string]any) []SSEEvent {
	var out []SSEEvent
	if u, ok := obj["usage"].(map[string]any); ok && len(u) > 0 {
		for k, v := range u {
			s.usage[k] = v
		}
	}
	choices, _ := obj["choices"].([]any)
	if len(choices) == 0 {
		return out
	}
	choice, _ := choices[0].(map[string]any)
	delta, _ := choice["delta"].(map[string]any)
	if delta == nil {
		return out
	}

	if think := reasoningText(delta); think != "" {
		if !s.reasoningOpen {
			s.reasoningOpen = true
			s.reasoningID = "rs_" + account.RandomHex(8)
			s.reasoningIdx = len(s.outputItems)
			// reasoning 项在 Codex 的历史回放里必须带 encrypted_content，
			// 否则下一轮请求会被序列化成缺字段的对象。上游没有真密文，
			// 这里给一个网关私有占位串；客户端只把它当不透明回放凭据。
			reasoningItem := map[string]any{
				"type": "reasoning", "id": s.reasoningID, "summary": []any{},
				"encrypted_content": "reasoning_" + account.RandomHex(16),
			}
			s.outputItems = append(s.outputItems, reasoningItem)
			out = append(out, SSEEvent{Name: "response.output_item.added", Data: map[string]any{
				"type": "response.output_item.added", "sequence_number": s.next(),
				"output_index": s.reasoningIdx,
				"item":         reasoningItem,
			}})
			out = append(out, SSEEvent{Name: "response.reasoning_summary_part.added", Data: map[string]any{
				"type": "response.reasoning_summary_part.added", "sequence_number": s.next(),
				"item_id": s.reasoningID, "output_index": s.reasoningIdx, "summary_index": 0,
				"part": map[string]any{"type": "summary_text", "text": ""},
			}})
		}
		s.reasoningTxt.WriteString(think)
		out = append(out, SSEEvent{Name: "response.reasoning_summary_text.delta", Data: map[string]any{
			"type": "response.reasoning_summary_text.delta", "sequence_number": s.next(),
			"item_id": s.reasoningID, "output_index": s.reasoningIdx, "summary_index": 0,
			"delta": think,
		}})
	}

	if text := str(delta["content"]); text != "" {
		if s.messageID == "" {
			s.messageID = "msg_" + account.RandomHex(8)
			s.textOpen = true
			s.outputItems = append(s.outputItems, map[string]any{
				"type": "message", "id": s.messageID, "role": "assistant",
				"status": "in_progress", "content": []any{},
			})
			out = append(out, SSEEvent{Name: "response.output_item.added", Data: map[string]any{
				"type": "response.output_item.added", "sequence_number": s.next(),
				"output_index": s.messageIndex(),
				"item": map[string]any{
					"type": "message", "id": s.messageID, "role": "assistant",
					"status": "in_progress", "content": []any{},
				},
			}})
			out = append(out, SSEEvent{Name: "response.content_part.added", Data: map[string]any{
				"type": "response.content_part.added", "sequence_number": s.next(),
				"item_id": s.messageID, "output_index": s.messageIndex(), "content_index": 0,
				"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
			}})
		}
		s.fullText.WriteString(text)
		out = append(out, SSEEvent{Name: "response.output_text.delta", Data: map[string]any{
			"type": "response.output_text.delta", "sequence_number": s.next(),
			"item_id": s.messageID, "output_index": s.messageIndex(), "content_index": 0,
			"delta": text,
		}})
	}

	if calls, ok := delta["tool_calls"].([]any); ok {
		for _, raw := range calls {
			call, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			idx := intOf(call["index"])
			acc, exists := s.tools[idx]
			if !exists {
				acc = &toolAccum{}
				s.tools[idx] = acc
				s.order = append(s.order, idx)
			}
			fn, _ := call["function"].(map[string]any)
			if id := str(call["id"]); id != "" {
				acc.ID = id
			}
			if name := str(fn["name"]); name != "" {
				acc.Name = name
			}
			// 参数只在这里写入一次。appended/缓冲的区分交给 appendToolDelta，
			// 由它决定“是否已经发出过 item.added”。
			out = append(out, s.appendToolDelta(idx, str(fn["arguments"]))...)
		}
	}
	if fr := str(choice["finish_reason"]); fr != "" {
		s.finished = true
	}
	return out
}

// appendToolDelta 把上游的增量参数转成 Responses 的 function_call 事件。
//
// 关键契约（codex-api/src/sse/responses.rs）：
//   - output_item.added 时 item 是 in_progress、arguments 为空；
//   - 参数用 response.function_call_arguments.delta 发，Codex 用它做 UI 增量；
//   - 最后 output_item.done 带完整 arguments。
//
// 旧实现把完整 item 直接塞进 added 且没有 delta，Codex 能“看到”调用但无法
// 正确把它绑定到本轮工具执行。
func (s *ResponsesStreamer) appendToolDelta(idx int, args string) []SSEEvent {
	var out []SSEEvent
	acc := s.tools[idx]
	if acc == nil {
		return out
	}

	// name 还没到：先缓存参数，但不能丢。等 name 到了再补发 added + 全量增量。
	if acc.Name == "" {
		acc.Args.WriteString(args)
		return out
	}

	if !s.emitted[idx] {
		s.emitted[idx] = true
		callID := acc.ID
		if callID == "" {
			callID = "call_" + account.RandomHex(6)
			acc.ID = callID
		}
		s.toolIndex[idx] = len(s.outputItems)
		s.toolCallIDs[idx] = callID
		itemID := "fc_" + account.RandomHex(8)
		s.toolIDs[idx] = itemID
		item := map[string]any{
			"type": "function_call", "id": itemID, "call_id": callID,
			"name": acc.Name, "arguments": "", "status": "in_progress",
		}
		s.outputItems = append(s.outputItems, item)
		s.toolCalls++
		out = append(out, SSEEvent{Name: "response.output_item.added", Data: map[string]any{
			"type": "response.output_item.added", "sequence_number": s.next(),
			"output_index": s.toolIndex[idx], "item": item,
		}})
		// 之前 name 未到时缓存的参数，以及本帧参数，一次性按增量补发。
		acc.Args.WriteString(args)
		if buffered := acc.Args.String(); buffered != "" {
			out = append(out, SSEEvent{Name: "response.function_call_arguments.delta", Data: map[string]any{
				"type": "response.function_call_arguments.delta", "sequence_number": s.next(),
				"item_id": itemID, "call_id": callID, "output_index": s.toolIndex[idx],
				"delta": buffered,
			}})
		}
		return out
	}

	if args != "" {
		acc.Args.WriteString(args)
		out = append(out, SSEEvent{Name: "response.function_call_arguments.delta", Data: map[string]any{
			"type": "response.function_call_arguments.delta", "sequence_number": s.next(),
			"item_id": s.toolIDs[idx], "call_id": s.toolCallIDs[idx],
			"output_index": s.toolIndex[idx],
			"delta":        args,
		}})
	}
	return out
}

// Usage 累积到的 usage（供记账）。
func (s *ResponsesStreamer) Usage() map[string]any { return s.usage }

// ToolCalls 本轮实际发出的工具调用数（供请求台账统计）。
func (s *ResponsesStreamer) ToolCalls() int { return s.toolCalls }

// Failure 发出 response.failed。上游在收口前断流时用它代替 response.completed，
// 避免客户端把一个半截 function_call 当成完整调用去执行。
func (s *ResponsesStreamer) Failure(code, message string) []SSEEvent {
	return []SSEEvent{{Name: "response.failed", Data: map[string]any{
		"type": "response.failed", "sequence_number": s.next(),
		"response": map[string]any{
			"id": s.ResponseID, "object": "response", "status": "failed",
			"model": s.Model, "output": s.outputItems, "usage": s.usageEnvelope(),
			"created_at": timeNow(), "incomplete_details": nil,
			"error": map[string]any{"code": code, "message": message},
		},
	}}}
}

// Finish 收尾：发 output_item.done 与 response.completed。
func (s *ResponsesStreamer) Finish() []SSEEvent {
	var out []SSEEvent

	if s.reasoningID != "" {
		summary := s.reasoningTxt.String()
		out = append(out, SSEEvent{Name: "response.reasoning_summary_text.done", Data: map[string]any{
			"type": "response.reasoning_summary_text.done", "sequence_number": s.next(),
			"item_id": s.reasoningID, "output_index": s.reasoningIdx, "summary_index": 0,
			"text": summary,
		}})
		reasoningItem := map[string]any{
			"type": "reasoning", "id": s.reasoningID,
			"summary":           []any{map[string]any{"type": "summary_text", "text": summary}},
			"encrypted_content": "reasoning_" + account.RandomHex(16),
		}
		out = append(out, SSEEvent{Name: "response.output_item.done", Data: map[string]any{
			"type": "response.output_item.done", "sequence_number": s.next(),
			"output_index": s.reasoningIdx,
			"item":         reasoningItem,
		}})
		s.outputItems[s.reasoningIdx] = reasoningItem
	}

	if s.messageID != "" {
		text := s.fullText.String()
		out = append(out, SSEEvent{Name: "response.output_text.done", Data: map[string]any{
			"type": "response.output_text.done", "sequence_number": s.next(),
			"item_id": s.messageID, "output_index": s.messageIndex(), "content_index": 0,
			"text": text,
		}})
		out = append(out, SSEEvent{Name: "response.content_part.done", Data: map[string]any{
			"type": "response.content_part.done", "sequence_number": s.next(),
			"item_id": s.messageID, "output_index": s.messageIndex(), "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": text, "annotations": []any{}},
		}})
		messageItem := map[string]any{
			"type": "message", "id": s.messageID, "role": "assistant",
			"status": "completed",
			"content": []any{map[string]any{
				"type": "output_text", "text": text, "annotations": []any{},
			}},
		}
		out = append(out, SSEEvent{Name: "response.output_item.done", Data: map[string]any{
			"type": "response.output_item.done", "sequence_number": s.next(),
			"output_index": s.messageIndex(),
			"item":         messageItem,
		}})
		s.outputItems[s.messageIndex()] = messageItem
	}

	for _, idx := range s.order {
		acc := s.tools[idx]
		if acc == nil || acc.Name == "" {
			continue
		}
		args := acc.Args.String()
		if args == "" {
			args = "{}"
		}
		// 参数不是完整 JSON 时不能发 done：那是上游断流留下的半截调用，
		// 发出去客户端必然解析失败。留给 handler 发 response.failed。
		if !json.Valid([]byte(args)) {
			continue
		}
		// 上游可能在收尾前才给 name；这时补发 added + 参数增量。
		if !s.emitted[idx] {
			s.emitted[idx] = true
			callID := acc.ID
			if callID == "" {
				callID = "call_" + account.RandomHex(6)
			}
			s.toolIndex[idx] = len(s.outputItems)
			s.toolCallIDs[idx] = callID
			itemID := "fc_" + account.RandomHex(8)
			s.toolIDs[idx] = itemID
			pending := map[string]any{
				"type": "function_call", "id": itemID, "call_id": callID,
				"name": acc.Name, "arguments": "", "status": "in_progress",
			}
			s.outputItems = append(s.outputItems, pending)
			s.toolCalls++
			out = append(out, SSEEvent{Name: "response.output_item.added", Data: map[string]any{
				"type": "response.output_item.added", "sequence_number": s.next(),
				"output_index": s.toolIndex[idx], "item": pending,
			}})
			out = append(out, SSEEvent{Name: "response.function_call_arguments.delta", Data: map[string]any{
				"type": "response.function_call_arguments.delta", "sequence_number": s.next(),
				"item_id": pending["id"], "call_id": callID,
				"output_index": s.toolIndex[idx], "delta": args,
			}})
		}
		callID := s.toolCallIDs[idx]
		if callID == "" {
			callID = acc.ID
			if callID == "" {
				callID = "call_" + account.RandomHex(6)
			}
			s.toolCallIDs[idx] = callID
		}
		pos := s.toolIndex[idx]
		id := s.toolIDs[idx]
		if prev, ok := s.outputItems[pos].(map[string]any); ok {
			if v := str(prev["id"]); v != "" {
				id = v
			}
			// 保留 added 时分配的下标；doned 用同一 index。
			if fr, ok := prev["status"]; ok && fr == "completed" {
				continue
			}
		}
		item := map[string]any{
			"type": "function_call", "id": id, "call_id": callID,
			"name": acc.Name, "arguments": args, "status": "completed",
		}
		out = append(out, SSEEvent{Name: "response.function_call_arguments.done", Data: map[string]any{
			"type": "response.function_call_arguments.done", "sequence_number": s.next(),
			"item_id": id, "call_id": callID,
			"output_index": pos, "arguments": args,
		}})
		out = append(out, SSEEvent{Name: "response.output_item.done", Data: map[string]any{
			"type": "response.output_item.done", "sequence_number": s.next(),
			"output_index": pos, "item": item,
		}})
		s.outputItems[pos] = item
	}

	out = append(out, SSEEvent{Name: "response.completed", Data: map[string]any{
		"type": "response.completed", "sequence_number": s.next(),
		"response": s.envelope("completed", s.outputItems),
	}})
	return out
}
