package main

import (
	"strings"
	"testing"

	"github.com/mattn/go-runewidth"
)

func TestWrapDisplayTextRespectsCJKCellWidth(t *testing.T) {
	lines := wrapDisplayText("你好世界abc", 6)
	want := []string{"你好世", "界abc"}
	if len(lines) != len(want) {
		t.Fatalf("expected %d lines, got %d: %#v", len(want), len(lines), lines)
	}
	for i := range want {
		if lines[i] != want[i] {
			t.Fatalf("line %d: expected %q, got %q", i, want[i], lines[i])
		}
		if width := runewidth.StringWidth(lines[i]); width > 6 {
			t.Fatalf("line %d exceeds width: %d", i, width)
		}
	}
}

func TestRenderViewportPadsToFullHeight(t *testing.T) {
	view := renderViewport([]string{"短"}, 6, 3)
	if got := strings.Count(view, "\n") + 1; got != 3 {
		t.Fatalf("expected padded viewport height 3, got %d", got)
	}
}
