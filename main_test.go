package main

import (
	"context"
	"testing"
)

func TestToolCallAccumulatorAssemblesStreamingFragments(t *testing.T) {
	acc := newToolCallAccumulator()
	acc.addDelta(0, "call_1", "function", "she", `{"command":"Get-`)
	acc.addDelta(0, "", "", "ll", `ChildItem"}`)

	calls := acc.calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].ID != "call_1" {
		t.Fatalf("unexpected id: %q", calls[0].ID)
	}
	if calls[0].Function.Name != "shell" {
		t.Fatalf("unexpected function name: %q", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments != `{"command":"Get-ChildItem"}` {
		t.Fatalf("unexpected arguments: %q", calls[0].Function.Arguments)
	}
}

func TestExecuteToolCallRejectsUnknownTool(t *testing.T) {
	result := executeToolCall(context.Background(), chatToolCall{
		ID:   "call_1",
		Type: "function",
		Function: chatToolCallFunc{
			Name:      "unknown",
			Arguments: `{}`,
		},
	})
	if result != "tool error: unsupported tool unknown" {
		t.Fatalf("unexpected result: %q", result)
	}
}
