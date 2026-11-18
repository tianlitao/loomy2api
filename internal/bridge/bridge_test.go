package bridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestToOpenAIPlainText 字符串 content 直接透传。
func TestToOpenAIPlainText(t *testing.T) {
	req := map[string]any{
		"model":    "imodel/spark-x",
		"system":   "be nice",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	out := ToOpenAI(req)
	msgs := out["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("期望 2 条消息（system + user），实际 %d", len(msgs))
	}
	first := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be nice" {
		t.Errorf("system 未正确翻译: %v", first)
	}
}

// TestToOpenAIToolRoundTrip tool_use / tool_result 双向翻译。
func TestToOpenAIToolRoundTrip(t *testing.T) {
	req := map[string]any{
		"model": "m",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{map[string]any{
					"type":  "tool_use",
					"id":    "toolu_1",
					"name":  "get_weather",
					"input": map[string]any{"city": "北京"},
				}},
			},
			map[string]any{
				"role": "user",
				"content": []any{map[string]any{
					"type":        "tool_result",
					"tool_use_id": "toolu_1",
					"content":     "晴 25°C",
				}},
			},
		},
	}
	out := ToOpenAI(req)
	msgs := out["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("期望 2 条消息，实际 %d: %v", len(msgs), msgs)
	}
	assistant := msgs[0].(map[string]any)
	calls, ok := assistant["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("assistant 应带 1 个 tool_call: %v", assistant)
	}
	call := calls[0].(map[string]any)
	if call["id"] != "toolu_1" {
		t.Errorf("tool_call id 未保留: %v", call)
	}
	fn := call["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("函数名未保留: %v", fn)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil {
		t.Fatalf("arguments 不是合法 JSON: %v", err)
	}
	if args["city"] != "北京" {
		t.Errorf("参数未保留: %v", args)
	}

	// tool_result 应变成独立的 tool 消息，且不产生空 user 脚手架
	toolMsg := msgs[1].(map[string]any)
	if toolMsg["role"] != "tool" {
		t.Errorf("tool_result 应翻译成 tool 消息，实际 %v", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "toolu_1" {
		t.Errorf("tool_call_id 未保留: %v", toolMsg)
	}
	if toolMsg["content"] != "晴 25°C" {
		t.Errorf("工具结果未保留: %v", toolMsg["content"])
	}
}

// TestToOpenAITools 工具定义翻译（Anthropic input_schema → parameters）。
func TestToOpenAITools(t *testing.T) {
	req := map[string]any{
		"model":    "m",
		"messages": []any{map[string]any{"role": "user", "content": "x"}},
		"tools": []any{map[string]any{
			"name":        "search",
			"description": "联网搜索",
			"input_schema": map[string]any{
				"type":       "object",
				"properties": map[string]any{"q": map[string]any{"type": "string"}},
			},
		}},
	}
	out := ToOpenAI(req)
	tools, ok := out["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools 未翻译: %v", out["tools"])
	}
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "search" || fn["description"] != "联网搜索" {
		t.Errorf("工具元信息丢失: %v", fn)
	}
	if fn["parameters"] == nil {
		t.Error("parameters 不应为空")
	}
}

// TestToOpenAIMaxTokens 关键参数必须原样带过去。
func TestToOpenAIMaxTokens(t *testing.T) {
	out := ToOpenAI(map[string]any{
		"model": "m", "max_tokens": 4096, "temperature": 0.7, "top_p": 0.9,
		"stream":         true,
		"messages":       []any{map[string]any{"role": "user", "content": "x"}},
		"stop_sequences": []any{"\n\n"},
	})
	if out["max_tokens"] != 4096 {
		t.Errorf("max_tokens 丢失: %v", out["max_tokens"])
	}
	if out["temperature"] != 0.7 || out["top_p"] != 0.9 {
		t.Error("采样参数丢失")
	}
	if out["stream"] != true {
		t.Error("stream 丢失")
	}
	if out["stop"] == nil {
		t.Error("stop_sequences 应映射成 stop")
	}
}

// TestToAnthropicText 非流式响应翻译。
func TestToAnthropicText(t *testing.T) {
	obj := map[string]any{
		"id": "chatcmpl-1", "model": "imodel/spark-x", "created": 1700000000,
		"choices": []any{map[string]any{
			"index": 0,
			"message": map[string]any{
				"role": "assistant", "content": "你好",
				"reasoning_content": "先想一想",
			},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 20, "points_consumed": 3},
	}
	out := ToAnthropic(obj, "imodel/spark-x")
	content := out["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("期望 2 个块（thinking + text），实际 %d: %v", len(content), content)
	}
	thinking := content[0].(map[string]any)
	if thinking["type"] != "thinking" || thinking["thinking"] != "先想一想" {
		t.Errorf("思考块不正确: %v", thinking)
	}
	text := content[1].(map[string]any)
	if text["type"] != "text" || text["text"] != "你好" {
		t.Errorf("文本块不正确: %v", text)
	}
	if out["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v", out["stop_reason"])
	}
	usage := out["usage"].(map[string]any)
	if usage["input_tokens"] != 10 || usage["output_tokens"] != 20 {
		t.Errorf("usage 未映射: %v", usage)
	}
	if usage["points_consumed"] != 3 {
		t.Error("points_consumed 应透出")
	}
}

// TestToAnthropicToolUse finish_reason=tool_calls 应映射为 tool_use。
func TestToAnthropicToolUse(t *testing.T) {
	obj := map[string]any{
		"choices": []any{map[string]any{
			"message": map[string]any{
				"role": "assistant", "content": nil,
				"tool_calls": []any{map[string]any{
					"id": "call_1", "type": "function",
					"function": map[string]any{"name": "f", "arguments": `{"a":1}`},
				}},
			},
			"finish_reason": "tool_calls",
		}},
	}
	out := ToAnthropic(obj, "m")
	if out["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v，期望 tool_use", out["stop_reason"])
	}
	block := out["content"].([]any)[0].(map[string]any)
	if block["type"] != "tool_use" || block["name"] != "f" {
		t.Errorf("工具块不正确: %v", block)
	}
	input := block["input"].(map[string]any)
	if input["a"] != float64(1) {
		t.Errorf("参数未解析: %v", input)
	}
}

// TestStopReasonMapping finish_reason 全覆盖。
func TestStopReasonMapping(t *testing.T) {
	cases := map[string]string{
		"length":        "max_tokens",
		"tool_calls":    "tool_use",
		"function_call": "tool_use",
		"stop":          "end_turn",
		"":              "end_turn",
	}
	for in, want := range cases {
		if got := StopReason(in); got != want {
			t.Errorf("StopReason(%q) = %q，期望 %q", in, got, want)
		}
	}
}

// TestStreamTranslatorText 流式文本增量翻译成 text 块。
func TestStreamTranslatorText(t *testing.T) {
	st := NewStreamTranslator("m", "msg_1")
	events := st.Start()
	if len(events) != 1 || events[0].Name != "message_start" {
		t.Fatalf("首个事件应是 message_start: %v", events)
	}

	var out []Event
	out = append(out, st.Feed(map[string]any{
		"choices": []any{map[string]any{
			"delta":         map[string]any{"content": "你"},
			"finish_reason": nil,
		}},
	})...)
	out = append(out, st.Feed(map[string]any{
		"choices": []any{map[string]any{
			"delta":         map[string]any{"content": "好"},
			"finish_reason": "stop",
		}},
	})...)
	out = append(out, st.Finish()...)

	var names []string
	var text strings.Builder
	for _, ev := range out {
		names = append(names, ev.Name)
		if ev.Name == "content_block_delta" {
			d := ev.Data["delta"].(map[string]any)
			if d["type"] == "text_delta" {
				text.WriteString(d["text"].(string))
			}
		}
	}
	if text.String() != "你好" {
		t.Errorf("聚合文本 = %q，期望 你好", text.String())
	}
	joined := strings.Join(names, ",")
	if !strings.Contains(joined, "content_block_start") {
		t.Error("缺少 content_block_start")
	}
	if !strings.Contains(joined, "message_delta") || !strings.Contains(joined, "message_stop") {
		t.Errorf("缺少收尾事件: %s", joined)
	}
	// content_block_start 只应出现一次（不能每帧都开新块）
	if strings.Count(joined, "content_block_start") != 1 {
		t.Errorf("content_block_start 次数 = %d，期望 1", strings.Count(joined, "content_block_start"))
	}
}

// TestStreamTranslatorThinking 思考链应先于文本开启，且切换时关闭前一块。
func TestStreamTranslatorThinking(t *testing.T) {
	st := NewStreamTranslator("m", "msg_1")
	var out []Event
	out = append(out, st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "想"}}},
	})...)
	out = append(out, st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": "答"}}},
	})...)
	out = append(out, st.Finish()...)

	var idxThinking, idxText = -1, -1
	stops := 0
	for _, ev := range out {
		switch ev.Name {
		case "content_block_start":
			block := ev.Data["content_block"].(map[string]any)
			if block["type"] == "thinking" && idxThinking < 0 {
				idxThinking = ev.Data["index"].(int)
			}
			if block["type"] == "text" && idxText < 0 {
				idxText = ev.Data["index"].(int)
			}
		case "content_block_stop":
			stops++
		}
	}
	if idxThinking < 0 || idxText < 0 {
		t.Fatalf("应同时出现 thinking 与 text 块: %d / %d", idxThinking, idxText)
	}
	if idxThinking >= idxText {
		t.Errorf("thinking 块应在 text 之前（index %d vs %d）", idxThinking, idxText)
	}
	if stops != 2 {
		t.Errorf("两个块都应收尾，实际 %d 次 stop", stops)
	}
}

// TestStreamTranslatorToolArgsAccumulated 分帧发送的工具参数要被拼接完整。
func TestStreamTranslatorToolArgsAccumulated(t *testing.T) {
	st := NewStreamTranslator("m", "msg_1")
	st.Start()
	all := st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{
				"index": 0, "id": "call_1",
				"function": map[string]any{"name": "search", "arguments": `{"q":`},
			}},
		}}},
	})
	all = append(all, st.Feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index":    0,
					"function": map[string]any{"arguments": `"天气"}`},
				}},
			},
			"finish_reason": "tool_calls",
		}},
	})...)
	events := append(all, st.Finish()...)

	var partial string
	var stopReason string
	for _, ev := range events {
		if ev.Name == "content_block_delta" {
			d := ev.Data["delta"].(map[string]any)
			if d["type"] == "input_json_delta" {
				partial += d["partial_json"].(string)
			}
		}
		if ev.Name == "message_delta" {
			d := ev.Data["delta"].(map[string]any)
			stopReason = d["stop_reason"].(string)
		}
	}
	if partial != `{"q":"天气"}` {
		t.Errorf("参数拼接 = %q", partial)
	}
	if stopReason != "tool_use" {
		t.Errorf("stop_reason = %q，期望 tool_use", stopReason)
	}
}

// TestResponsesToChatMessage 基础 input 翻译。
func TestResponsesToChatMessage(t *testing.T) {
	out, err := ResponsesToChat(map[string]any{
		"model":        "m",
		"instructions": "你是助手",
		"input":        "你好",
	})
	if err != nil {
		t.Fatalf("翻译失败: %v", err)
	}
	msgs := out["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("期望 2 条消息，实际 %d", len(msgs))
	}
	if msgs[0].(map[string]any)["role"] != "system" {
		t.Error("instructions 应变成 system")
	}
	if msgs[1].(map[string]any)["content"] != "你好" {
		t.Error("input 字符串应变成 user 消息")
	}
}

// TestResponsesToChatFunctionCall function_call / output 条目翻译。
func TestResponsesToChatFunctionCall(t *testing.T) {
	out, err := ResponsesToChat(map[string]any{
		"model": "m",
		"input": []any{
			map[string]any{"type": "message", "role": "user",
				"content": []any{map[string]any{"type": "input_text", "text": "查天气"}}},
			map[string]any{"type": "function_call", "call_id": "call_1",
				"name": "get_weather", "arguments": `{"city":"北京"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_1",
				"output": "晴"},
		},
	})
	if err != nil {
		t.Fatalf("翻译失败: %v", err)
	}
	msgs := out["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("期望 3 条消息，实际 %d", len(msgs))
	}
	assistant := msgs[1].(map[string]any)
	calls := assistant["tool_calls"].([]any)
	if len(calls) != 1 || calls[0].(map[string]any)["id"] != "call_1" {
		t.Errorf("function_call 未正确翻译: %v", assistant)
	}
	tool := msgs[2].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_1" {
		t.Errorf("function_call_output 未正确翻译: %v", tool)
	}
}

// TestResponsesToChatEmptyInput 空 input 必须报错而非产生空请求。
func TestResponsesToChatEmptyInput(t *testing.T) {
	if _, err := ResponsesToChat(map[string]any{"model": "m"}); err == nil {
		t.Error("空 input 应返回错误")
	}
}

// TestChatToResponse 非流式响应翻译。
func TestChatToResponse(t *testing.T) {
	obj := map[string]any{
		"id": "chatcmpl-1", "model": "m", "created": 1700000000,
		"choices": []any{map[string]any{
			"message":       map[string]any{"role": "assistant", "content": "答"},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 7},
	}
	out := ChatToResponse(obj, "m")
	if out["object"] != "response" || out["status"] != "completed" {
		t.Errorf("响应对象形态不对: %v", out)
	}
	output := out["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("期望 1 个 output 条目，实际 %d", len(output))
	}
	msg := output[0].(map[string]any)
	if msg["type"] != "message" {
		t.Errorf("output 类型 = %v", msg["type"])
	}
	content := msg["content"].([]any)[0].(map[string]any)
	if content["text"] != "答" {
		t.Errorf("文本丢失: %v", content)
	}
	usage := out["usage"].(map[string]any)
	if usage["input_tokens"] != 5 || usage["output_tokens"] != 7 {
		t.Errorf("usage 未映射: %v", usage)
	}
	if usage["total_tokens"] != 12 {
		t.Errorf("total_tokens = %v，期望 12", usage["total_tokens"])
	}
}

// TestResponsesStreamerSequence 流式事件序列完整（created → delta → completed）。
func TestResponsesStreamerSequence(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_1")
	var names []string
	for _, ev := range s.Start() {
		names = append(names, ev.Name)
	}
	for _, ev := range s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": "你好"}}},
	}) {
		names = append(names, ev.Name)
	}
	for _, ev := range s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{}, "finish_reason": "stop"}},
	}) {
		names = append(names, ev.Name)
	}
	for _, ev := range s.Finish() {
		names = append(names, ev.Name)
	}

	joined := strings.Join(names, ",")
	for _, want := range []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.done",
		"response.completed",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("缺少事件 %s，实际序列: %s", want, joined)
		}
	}
	// sequence_number 必须单调递增
	seq := -1
	for _, ev := range s.Finish() {
		if n, ok := ev.Data["sequence_number"].(int); ok {
			if n <= seq {
				t.Errorf("sequence_number 未递增: %d after %d", n, seq)
			}
			seq = n
		}
	}
}

// TestResponsesStreamCompletedUsage Codex 的 ResponseCompletedUsage 对 usage 做严格
// 反序列化：input_tokens / output_tokens / total_tokens 缺一个，整轮在工具执行后
// 仍会以 “missing field total_tokens” 失败并重试。
func TestResponsesStreamCompletedUsage(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_1")
	s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": "ok"}}},
		"usage": map[string]any{
			"prompt_tokens": 12, "completion_tokens": 7,
			"prompt_tokens_details":     map[string]any{"cached_tokens": 8},
			"completion_tokens_details": map[string]any{"reasoning_tokens": 5},
		},
	})
	var completed map[string]any
	for _, ev := range s.Finish() {
		if ev.Name == "response.completed" {
			completed = ev.Data["response"].(map[string]any)
		}
	}
	if completed == nil {
		t.Fatal("缺少 response.completed")
	}
	usage := completed["usage"].(map[string]any)
	for _, k := range []string{"input_tokens", "output_tokens", "total_tokens"} {
		if _, ok := usage[k]; !ok {
			t.Fatalf("usage 缺字段 %s: %v", k, usage)
		}
	}
	if usage["total_tokens"] != 19 {
		t.Errorf("total_tokens = %v，期望 19", usage["total_tokens"])
	}
	if d := usage["output_tokens_details"].(map[string]any); d["reasoning_tokens"] != 5 {
		t.Errorf("reasoning_tokens = %v，期望 5", d["reasoning_tokens"])
	}
}

// TestResponsesStreamToolEventContract Codex 依赖 output_item.added(in_progress,
// arguments="") + function_call_arguments.delta + output_item.done(completed)。
// 旧实现只在收尾发一个 completed 整块，客户端拿不到可绑定的调用项。
func TestResponsesStreamToolEventContract(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_1")
	var events []SSEEvent
	events = append(events, s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{
				"index": 0, "id": "call_1",
				"function": map[string]any{"name": "get_weather", "arguments": "{"},
			}},
		}}},
	})...)
	events = append(events, s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{
				"index":    0,
				"function": map[string]any{"arguments": `"city":"北京"}`},
			}},
		}, "finish_reason": "tool_calls"}},
	})...)
	events = append(events, s.Finish()...)

	var deltaCount, addedPos, donePos int
	var sawAdded, sawDone, sawDelta bool
	var addedStatus, addedArgs, addedID string
	var doneStatus, doneArgs, doneID, callID string
	var deltas []string
	for i, ev := range events {
		switch ev.Name {
		case "response.output_item.added":
			item := ev.Data["item"].(map[string]any)
			if item["type"] != "function_call" {
				continue
			}
			sawAdded = true
			_ = i
			addedPos = ev.Data["output_index"].(int)
			addedStatus = item["status"].(string)
			addedArgs = item["arguments"].(string)
			addedID = item["id"].(string)
			callID = item["call_id"].(string)
		case "response.function_call_arguments.delta":
			sawDelta = true
			deltaCount++
			if ev.Data["item_id"] != addedID {
				t.Errorf("delta item_id=%v，added item_id=%s", ev.Data["item_id"], addedID)
			}
			deltas = append(deltas, ev.Data["delta"].(string))
		case "response.output_item.done":
			item := ev.Data["item"].(map[string]any)
			if item["type"] != "function_call" {
				continue
			}
			sawDone = true
			donePos = ev.Data["output_index"].(int)
			doneStatus = item["status"].(string)
			doneArgs = item["arguments"].(string)
			doneID = item["id"].(string)
		}
	}
	if !sawAdded || !sawDone || !sawDelta {
		for i, ev := range events {
			t.Logf("%d %s %v", i, ev.Name, ev.Data)
		}
		t.Fatal("function_call 的 added/done 事件不完整")
	}
	if addedStatus != "in_progress" || addedArgs != "" {
		t.Errorf("added 应为 in_progress 空参数，实际 %s / %q", addedStatus, addedArgs)
	}
	if doneStatus != "completed" || doneArgs != `{"city":"北京"}` {
		t.Errorf("done 参数不完整: %s / %q", doneStatus, doneArgs)
	}
	if addedID != doneID || doneID == "" {
		t.Errorf("added/done item_id 不一致: %q vs %q", addedID, doneID)
	}
	if callID == "" || callID != "call_1" {
		t.Errorf("call_id 未保留: %q", callID)
	}
	if addedPos != donePos {
		t.Errorf("added/done output_index 不一致: %d vs %d", addedPos, donePos)
	}
	if deltaCount != 2 || strings.Join(deltas, "") != `{"city":"北京"}` {
		t.Errorf("参数增量 = %v（%d 帧）", deltas, deltaCount)
	}
}

// TestResponsesStreamTruncated 上游不发 finish_reason 就断流时，半截工具参数
// 必须被识别为不完整，且 Finish 不能把它标成 completed 发出去。
func TestResponsesStreamTruncated(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_t")
	s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{
				"index": 0, "id": "call_t",
				"function": map[string]any{"name": "shell", "arguments": `{"cmd":"git st`},
			}},
		}}},
	})
	if !s.Truncated() {
		t.Fatal("半截参数 + 无 finish_reason 应判为截断")
	}
	var doneArgs string
	for _, ev := range s.Finish() {
		if ev.Name == "response.function_call_arguments.done" {
			doneArgs = ev.Data["arguments"].(string)
		}
	}
	if doneArgs != "" {
		t.Errorf("截断的调用不应发 arguments.done，实际 %q", doneArgs)
	}
}

// TestResponsesStreamComplete 正常 finish_reason 的完整参数不应误判截断。
func TestResponsesStreamComplete(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_c")
	s.Feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index": 0, "id": "call_c",
					"function": map[string]any{"name": "shell", "arguments": `{"cmd":"pwd"}`},
				}},
			},
			"finish_reason": "tool_calls",
		}},
	})
	if s.Truncated() {
		t.Fatal("完整参数 + finish_reason 不应判截断")
	}
}

// TestAnthropicStreamTruncated Anthropic 侧同样不能把断流当正常收尾。
func TestAnthropicStreamTruncated(t *testing.T) {
	s := NewStreamTranslator("m", "msg_t")
	s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{
				"index": 0, "id": "toolu_t",
				"function": map[string]any{"name": "shell", "arguments": `{"cmd":"ec`},
			}},
		}}},
	})
	if !s.Truncated() {
		t.Fatal("Anthropic 半截参数 + 无 finish_reason 应判为截断")
	}
}

// TestResponsesStreamReasoningSummary 思考链要真实进入 summary，而不是空串占位；
// 且 reasoning 项必须带 encrypted_content，供 Codex 下一轮回放。
func TestResponsesStreamReasoningSummary(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_1")
	var events []SSEEvent
	events = append(events, s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "先"}}},
	})...)
	events = append(events, s.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "想"}}},
	})...)
	events = append(events, s.Finish()...)

	var deltaText, doneText, summary string
	var encrypted string
	for _, ev := range events {
		switch ev.Name {
		case "response.reasoning_summary_text.delta":
			deltaText += ev.Data["delta"].(string)
		case "response.reasoning_summary_text.done":
			doneText = ev.Data["text"].(string)
		case "response.output_item.done":
			item := ev.Data["item"].(map[string]any)
			if item["type"] != "reasoning" {
				continue
			}
			encrypted, _ = item["encrypted_content"].(string)
			parts := item["summary"].([]any)
			if len(parts) > 0 {
				summary = parts[0].(map[string]any)["text"].(string)
			}
		}
	}
	if deltaText != "先想" || doneText != "先想" || summary != "先想" {
		t.Errorf("思考摘要丢失: delta=%q done=%q summary=%q", deltaText, doneText, summary)
	}
	if encrypted == "" {
		t.Error("reasoning 项缺 encrypted_content")
	}
}

// TestResponsesToChatReasoning 请求侧 reasoning.effort 不能被静默丢弃。
func TestResponsesToChatReasoning(t *testing.T) {
	out, err := ResponsesToChat(map[string]any{
		"model":     "m",
		"input":     "hi",
		"reasoning": map[string]any{"effort": "high"},
	})
	if err != nil {
		t.Fatalf("翻译失败: %v", err)
	}
	if out["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v，期望 high", out["reasoning_effort"])
	}

	out, _ = ResponsesToChat(map[string]any{
		"model": "m", "input": "hi",
		"reasoning": map[string]any{"effort": "none"},
	})
	if out["enable_thinking"] != false {
		t.Errorf("effort=none 应透传关闭开关，实际 %v", out["enable_thinking"])
	}
}

// TestResponsesStreamParallelToolArgs 并行调用时两个 index 的参数必须按各自的
// call_id 完整到达，且流式增量拼接 == done 里的完整参数。
//
// 真实故障（2026-11-18）：上游同轮返回两个 exec_command，Codex 记录的 done
// 参数少了收尾的 "}"，工具执行直接失败。这里把“name 与首段参数同帧 + 后续帧
// 只带参数 + 尾帧只有单个 }”的真实形态固定下来。
func TestResponsesStreamParallelToolArgs(t *testing.T) {
	s := NewResponsesStreamer("m", "resp_1")
	var events []SSEEvent

	// 上游帧：两路并行，首帧 name + 首段参数，随后两次纯参数增量，最后单 }。
	type argFrame struct {
		index int
		id    string
		name  string
		args  string
	}
	frames := [][]argFrame{
		{
			{0, "call_a", "exec_command", `{"cmd":"git status --short`},
			{1, "call_b", "exec_command", `{"cmd":"ls internal/bridge`},
		},
		{{0, "", "", "\""}, {1, "", "", "\""}},
		{
			{0, "", "", "}"},
			{1, "", "", "}"},
		},
	}
	for _, frame := range frames {
		calls := make([]any, 0, len(frame))
		for _, f := range frame {
			fn := map[string]any{"arguments": f.args}
			if f.name != "" {
				fn["name"] = f.name
			}
			call := map[string]any{"index": f.index, "function": fn}
			if f.id != "" {
				call["id"] = f.id
			}
			calls = append(calls, call)
		}
		events = append(events, s.Feed(map[string]any{"choices": []any{map[string]any{
			"delta": map[string]any{"tool_calls": calls},
		}}})...)
	}

	streamed := map[string]string{}
	for _, ev := range events {
		if ev.Name == "response.function_call_arguments.delta" {
			streamed[ev.Data["call_id"].(string)] += ev.Data["delta"].(string)
		}
	}
	done := map[string]string{}
	for _, ev := range s.Finish() {
		if ev.Name == "response.function_call_arguments.done" {
			done[ev.Data["call_id"].(string)] = ev.Data["arguments"].(string)
		}
		if ev.Name == "response.output_item.done" {
			item := ev.Data["item"].(map[string]any)
			if item["type"] == "function_call" {
				if got := item["arguments"].(string); got != done[item["call_id"].(string)] {
					t.Errorf("output_item.done 与 arguments.done 不一致: %q vs %q", got, done[item["call_id"].(string)])
				}
			}
		}
	}

	want := map[string]string{
		"call_a": `{"cmd":"git status --short"}`,
		"call_b": `{"cmd":"ls internal/bridge"}`,
	}
	for callID, exp := range want {
		if done[callID] != exp {
			t.Errorf("%s done 参数 = %q，期望 %q", callID, done[callID], exp)
		}
		if streamed[callID] != exp {
			t.Errorf("%s 增量拼接 = %q，期望 %q", callID, streamed[callID], exp)
		}
	}
}

// TestAnthropicStreamThinkingSignature thinking 块必须以 signature_delta 收口，
// 否则 Claude 客户端下一轮回放会丢弃思考上下文。
func TestAnthropicStreamThinkingSignature(t *testing.T) {
	st := NewStreamTranslator("m", "msg_1")
	var events []Event
	events = append(events, st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "想一想"}}},
	})...)
	events = append(events, st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": "答案"}}},
	})...)
	events = append(events, st.Finish()...)

	var sig string
	var thinkClosed bool
	for _, ev := range events {
		if ev.Name == "content_block_delta" {
			d := ev.Data["delta"].(map[string]any)
			if d["type"] == "signature_delta" {
				sig = d["signature"].(string)
			}
		}
	}
	if sig == "" {
		t.Fatal("thinking 块缺 signature_delta")
	}
	_ = thinkClosed
}

// TestAnthropicStreamToolArgsStreamed 工具参数要在流中按 input_json_delta 增量
// 发出，且拼接结果完整；旧实现只在 Finish 整块发一次。
func TestAnthropicStreamToolArgsStreamed(t *testing.T) {
	st := NewStreamTranslator("m", "msg_1")
	var events []Event
	events = append(events, st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{"index": 0, "id": "call_1",
				"function": map[string]any{"name": "shell", "arguments": `{"cmd":"ls`}}},
		}}},
	})...)
	events = append(events, st.Feed(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{"index": 0,
				"function": map[string]any{"arguments": `"}`}}},
		}}},
	})...)
	events = append(events, st.Finish()...)

	var partial string
	var stops int
	for _, ev := range events {
		switch ev.Name {
		case "content_block_delta":
			d := ev.Data["delta"].(map[string]any)
			if d["type"] == "input_json_delta" {
				partial += d["partial_json"].(string)
			}
		case "content_block_stop":
			stops++
		}
	}
	if partial != `{"cmd":"ls"}` {
		t.Errorf("工具参数拼接 = %q", partial)
	}
	if stops == 0 {
		t.Error("缺 content_block_stop")
	}
}
