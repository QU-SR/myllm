//go:build !windows

package main

import (
	"context"
	"os/exec"
)

func configureDetachedProcess(cmd *exec.Cmd) {}

func configureLauncherProcess(cmd *exec.Cmd) {}

func runTray(layout Layout, baseURL string) error {
	<-context.Background().Done()
	return nil
}

func requestTrayQuit() {}
