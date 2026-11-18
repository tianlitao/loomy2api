package bridge

import "strings"

// reasoning 兼容层。
//
// Loomy 上游的 Chat Completions 目前以 reasoning_content 返回思考链，但同一个
// 网关在换模型、换客户端时还会出现 reasoning / reasoning_details /
// thinking_content 等字段。客户端侧（Codex、Claude Code）只认自己协议的思考块，
// 这里把各种上游变体归一成两样东西：
//   - 思考正文：用于 Anthropic thinking 块与 Responses reasoning summary；
//   - 思考开关与档位：用于把客户端请求翻译回上游真正认识的参数。
//
// 注意不要“看见 reasoning 字段就开思考”。有些客户端（包括 Codex 的历史回放）
// 会把 reasoning 块作为上下文回传，那只是历史，不是本轮要求。

// reasoningText 从一个 delta / message 对象里取思考正文。
//
// 顺序按上游真实抓包来的字段优先：reasoning_content 是 Loomy 当前使用的字段；
// reasoning 既可能是纯字符串，也可能是 {content|text} 对象；reasoning_details
// 是部分 OpenAI 兼容网关的数组形态。
func reasoningText(obj map[string]any) string {
	if obj == nil {
		return ""
	}
	if s := str(obj["reasoning_content"]); s != "" {
		return s
	}
	if s := str(obj["thinking_content"]); s != "" {
		return s
	}
	switch r := obj["reasoning"].(type) {
	case string:
		return r
	case map[string]any:
		if s := str(r["content"]); s != "" {
			return s
		}
		if s := str(r["text"]); s != "" {
			return s
		}
	}
	if details, ok := obj["reasoning_details"].([]any); ok {
		var b strings.Builder
		for _, raw := range details {
			d, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if s := str(d["text"]); s != "" {
				b.WriteString(s)
				continue
			}
			if s := str(d["content"]); s != "" {
				b.WriteString(s)
			}
		}
		return b.String()
	}
	return ""
}

// applyReasoning 把客户端协议的思考要求翻译成上游字段。
//
// 上游用 enable_thinking 开关思考，并接受 chat_template_kwargs 的同名参数。
// 实测 DeepSeek-V4.1-Flash 对 none/low/high 照样返回 reasoning_content，
// enable_thinking=false 也没能归零；所以这里只负责把客户端的意愿透传，
// 不假设上游一定服从。档位本身仍原样上送，避免网关吞掉客户端参数。
func applyReasoning(out map[string]any, req map[string]any) {
	if out == nil || req == nil {
		return
	}

	// Anthropic: {thinking: {type: "enabled"|"disabled"|"adaptive", budget_tokens}}
	if thinking, ok := req["thinking"].(map[string]any); ok {
		kind := strings.ToLower(str(thinking["type"]))
		switch kind {
		case "enabled", "adaptive":
			out["enable_thinking"] = true
		case "disabled":
			out["enable_thinking"] = false
			out["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
		}
	}

	// OpenAI 兼容: effort 原样透传；none/off 额外发关闭开关，由上游决定是否服从。
	effort := strings.ToLower(strings.TrimSpace(str(req["reasoning_effort"])))
	if effort == "" {
		if r, ok := req["reasoning"].(map[string]any); ok {
			effort = strings.ToLower(strings.TrimSpace(str(r["effort"])))
		}
	}
	if effort != "" {
		out["reasoning_effort"] = effort
		if effort == "none" || effort == "off" || effort == "disabled" {
			out["enable_thinking"] = false
			out["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
		}
	}

	// 部分客户端用 chat_template_kwargs 直接控制 Qwen 系模板；原样保留。
	if ctk, ok := req["chat_template_kwargs"].(map[string]any); ok && len(ctk) > 0 {
		out["chat_template_kwargs"] = ctk
	}
}
