package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type Layout struct {
	Root, ConfigDir, ModelsDir, BackendsDir  string
	SystemPrompt, ConfigFile, BackendExePath string
}

type Config struct {
	DefaultThreads        int     `json:"default_threads"`
	DefaultContextLength  int     `json:"default_context_length"`
	DefaultTemperature    float64 `json:"default_temperature"`
	DefaultTopP           float64 `json:"default_top_p"`
	DefaultModel          string  `json:"default_model"`
	PerformanceProfile    string  `json:"performance_profile"`
	KeepModelResident     bool    `json:"keep_model_resident"`
	ResidentIdleMinutes   int     `json:"resident_idle_minutes"`
	EnableMMProj          bool    `json:"enable_mmproj"`
	EnableShellTool       bool    `json:"enable_shell_tool"`
	ServerStartupSeconds  int     `json:"server_startup_seconds"`
	RequestTimeoutSeconds int     `json:"request_timeout_seconds"`
	MaxImageBytes         int64   `json:"max_image_bytes"`
	MaxFileBytes          int64   `json:"max_file_bytes"`
	DefaultSystemPrompt   string  `json:"default_system_prompt"`
}

type LocalModel struct {
	Name           string
	LocalPath      string
	MMProjPath     string
	SupportsVision bool
}

type Image struct {
	Path, MIMEType, Base64, DataURL string
	SizeBytes                       int64
}

type chatReq struct {
	Model       string           `json:"model"`
	Messages    []map[string]any `json:"messages"`
	Stream      bool             `json:"stream"`
	Temperature float64          `json:"temperature,omitempty"`
	TopP        float64          `json:"top_p,omitempty"`
	ImageData   []map[string]any `json:"image_data,omitempty"`
}

type ChatSession struct {
	ModelName     string
	History       []map[string]string
	SystemFile    string
	Threads       int
	ContextLength int
	Temperature   float64
	TopP          float64
}

type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

type DownloadProgress struct {
	Label      string
	Downloaded int64
	Total      int64
	SpeedBytes float64
	Done       bool
}

type PullResult struct {
	ModelName string
	Dest      string
}

type residentServerManager struct {
	mu        sync.Mutex
	cmd       *exec.Cmd
	baseURL   string
	key       string
	inUse     int
	idleTimer *time.Timer
}

var sharedServer residentServerManager

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	defer sharedServer.shutdown()
	if err := run(ctx, os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string) error {
	enableWindowsUTF8Console()

	layout, err := resolveLayout()
	if err != nil {
		return err
	}
	if err := ensureLayout(layout); err != nil {
		return err
	}
	cfg, err := loadConfig(layout)
	if err != nil {
		return err
	}
	if err := ensurePrompt(layout); err != nil {
		return err
	}

	if len(args) == 0 {
		return cmdChat(ctx, nil, layout, cfg)
	}

	switch args[0] {
	case "chat":
		return cmdChat(ctx, args[1:], layout, cfg)
	case "pull":
		return cmdPull(args[1:], layout)
	case "ls":
		return cmdList(layout)
	case "info":
		return cmdInfo(args[1:], layout)
	case "rm":
		return cmdRemove(args[1:], layout)
	case "doctor":
		return cmdDoctor(layout)
	case "bench":
		return cmdBench()
	case "run":
		return cmdRun(ctx, args[1:], layout, cfg)
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func usage() {
	fmt.Print(`myllm - portable local AI CLI

Commands:
  chat [model]
  pull <name-or-url> [--url URL | --path FILE]
  ls
  info <name>
  rm <name> [--delete-files]
  doctor
  bench
  run <name> [prompt text] [--image FILE] [--system-file FILE] [--interactive]

No arguments:
  Start interactive chat UI
`)
}

func enableWindowsUTF8Console() {
	if runtime.GOOS != "windows" {
		return
	}
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	const utf8CodePage = 65001
	_, _, _ = kernel32.NewProc("SetConsoleOutputCP").Call(uintptr(utf8CodePage))
	_, _, _ = kernel32.NewProc("SetConsoleCP").Call(uintptr(utf8CodePage))
}

func resolveLayout() (Layout, error) {
	root := os.Getenv("PORTABLELLM_HOME")
	if root == "" {
		exe, err := os.Executable()
		if err != nil {
			return Layout{}, err
		}
		root = filepath.Dir(exe)
	}
	root, _ = filepath.Abs(root)
	name := "llama-server"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return Layout{
		Root:           root,
		ConfigDir:      filepath.Join(root, "config"),
		ModelsDir:      filepath.Join(root, "models"),
		BackendsDir:    filepath.Join(root, "backends"),
		SystemPrompt:   filepath.Join(root, "config", "system_prompt.txt"),
		ConfigFile:     filepath.Join(root, "config", "config.json"),
		BackendExePath: filepath.Join(root, "backends", name),
	}, nil
}

func ensureLayout(l Layout) error {
	for _, dir := range []string{l.ConfigDir, l.BackendsDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		DefaultThreads:        4,
		DefaultContextLength:  4096,
		DefaultTemperature:    0.7,
		DefaultTopP:           0.95,
		DefaultModel:          "",
		PerformanceProfile:    "balanced",
		KeepModelResident:     false,
		ResidentIdleMinutes:   15,
		EnableMMProj:          true,
		EnableShellTool:       false,
		ServerStartupSeconds:  45,
		RequestTimeoutSeconds: 300,
		MaxImageBytes:         20 * 1024 * 1024,
		MaxFileBytes:          1 * 1024 * 1024,
		DefaultSystemPrompt:   filepath.Join("config", "system_prompt.txt"),
	}
}

func loadConfig(layout Layout) (Config, error) {
	cfg := defaultConfig()
	if _, err := os.Stat(layout.ConfigFile); os.IsNotExist(err) {
		if err := saveConfig(layout, cfg); err != nil {
			return Config{}, err
		}
		return cfg, nil
	}
	data, err := os.ReadFile(layout.ConfigFile)
	if err != nil {
		return Config{}, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}
	if !filepath.IsAbs(cfg.DefaultSystemPrompt) {
		cfg.DefaultSystemPrompt = filepath.Join(layout.Root, cfg.DefaultSystemPrompt)
	}
	return cfg, nil
}

func saveConfig(layout Layout, cfg Config) error {
	out := cfg
	if filepath.IsAbs(out.DefaultSystemPrompt) {
		if rel, err := filepath.Rel(layout.Root, out.DefaultSystemPrompt); err == nil {
			out.DefaultSystemPrompt = rel
		}
	}
	data, _ := json.MarshalIndent(out, "", "  ")
	return os.WriteFile(layout.ConfigFile, append(data, '\n'), 0o644)
}

func ensurePrompt(layout Layout) error {
	if _, err := os.Stat(layout.SystemPrompt); err == nil {
		return nil
	}
	return os.WriteFile(layout.SystemPrompt, []byte("You are a concise local assistant. Be accurate, practical, and transparent about limitations.\n"), 0o644)
}

func scanModels(layout Layout) ([]LocalModel, error) {
	entries, err := os.ReadDir(layout.ModelsDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	models := make([]LocalModel, 0)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		lower := strings.ToLower(name)
		if !strings.HasSuffix(lower, ".gguf") || strings.Contains(lower, "mmproj") {
			continue
		}
		full := filepath.Join(layout.ModelsDir, name)
		base := strings.TrimSuffix(name, filepath.Ext(name))
		modelName := decodeModelName(base)
		mmproj := findMMProj(layout.ModelsDir, base)
		models = append(models, LocalModel{
			Name:           modelName,
			LocalPath:      full,
			MMProjPath:     mmproj,
			SupportsVision: mmproj != "" || looksLikeVisionModel(modelName),
		})
	}
	sort.Slice(models, func(i, j int) bool { return strings.ToLower(models[i].Name) < strings.ToLower(models[j].Name) })
	return models, nil
}

func getModel(layout Layout, name string) (LocalModel, error) {
	models, err := scanModels(layout)
	if err != nil {
		return LocalModel{}, err
	}
	for _, model := range models {
		if strings.EqualFold(model.Name, name) {
			return model, nil
		}
		base := strings.TrimSuffix(filepath.Base(model.LocalPath), filepath.Ext(model.LocalPath))
		if strings.EqualFold(base, name) {
			return model, nil
		}
	}
	return LocalModel{}, errors.New("model not found")
}

func findMMProj(modelsDir, base string) string {
	candidates := []string{
		base + ".mmproj.gguf",
		base + "-mmproj.gguf",
		base + "_mmproj.gguf",
		"mmproj-" + base + ".gguf",
		"mmproj_" + base + ".gguf",
	}
	for _, candidate := range candidates {
		full := filepath.Join(modelsDir, candidate)
		if _, err := os.Stat(full); err == nil {
			return full
		}
	}
	return ""
}

func looksLikeVisionModel(name string) bool {
	value := strings.ToLower(name)
	for _, token := range []string{"vision", "vl", "llava", "qwen2.5vl", "qwen-vl"} {
		if strings.Contains(value, token) {
			return true
		}
	}
	return false
}

func cmdPull(args []string, layout Layout) error {
	lastLen := 0
	lastStage := ""
	progressPrinter := func(p DownloadProgress) {
		stage := strings.TrimSpace(p.Label)
		if stage != "" && stage != lastStage {
			switch stage {
			case "主模型":
				fmt.Println("正在下载主模型...")
			case "mmproj":
				fmt.Println("正在下载 mmproj...")
			}
			lastStage = stage
		}
		line := formatDownloadProgress(p)
		if len(line) < lastLen {
			line += strings.Repeat(" ", lastLen-len(line))
		}
		lastLen = len(line)
		fmt.Printf("\r%s", line)
		if p.Done {
			fmt.Println()
		}
	}
	result, err := pullModel(args, layout, progressPrinter)
	if err != nil {
		return err
	}
	if result.Dest != "" {
		fmt.Printf("Downloaded model %s to %s\n", result.ModelName, result.Dest)
	}
	return nil
}

func pullModel(args []string, layout Layout, progress func(DownloadProgress)) (PullResult, error) {
	source := ""
	parseArgs := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		source = args[0]
		parseArgs = args[1:]
	}

	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	urlFlag := fs.String("url", "", "GGUF download URL")
	pathFlag := fs.String("path", "", "local GGUF file")
	if err := fs.Parse(parseArgs); err != nil {
		return PullResult{}, err
	}
	if source == "" && fs.NArg() > 0 {
		source = fs.Arg(0)
	}
	if source == "" && *urlFlag == "" && *pathFlag == "" {
		return PullResult{}, errors.New("usage: myllm pull <name-or-url> [--url URL | --path FILE]")
	}

	if err := os.MkdirAll(layout.ModelsDir, 0o755); err != nil {
		return PullResult{}, err
	}

	switch {
	case *pathFlag != "":
		dst, err := copyModelInto(layout.ModelsDir, *pathFlag)
		if err != nil {
			return PullResult{}, err
		}
		return PullResult{ModelName: decodeModelName(strings.TrimSuffix(filepath.Base(dst), filepath.Ext(dst))), Dest: dst}, nil
	case *urlFlag != "":
		dst := filepath.Join(layout.ModelsDir, guessName(*urlFlag))
		if err := downloadToFile(*urlFlag, dst, guessName(*urlFlag), progress); err != nil {
			return PullResult{}, err
		}
		return PullResult{ModelName: decodeModelName(strings.TrimSuffix(filepath.Base(dst), filepath.Ext(dst))), Dest: dst}, nil
	case looksLikeURL(source):
		dst := filepath.Join(layout.ModelsDir, guessName(source))
		if err := downloadToFile(source, dst, guessName(source), progress); err != nil {
			return PullResult{}, err
		}
		return PullResult{ModelName: decodeModelName(strings.TrimSuffix(filepath.Base(dst), filepath.Ext(dst))), Dest: dst}, nil
	default:
		dst := filepath.Join(layout.ModelsDir, encodeModelFilename(source)+".gguf")
		if err := downloadNamedModel(source, dst, progress); err != nil {
			return PullResult{}, err
		}
		return PullResult{ModelName: source, Dest: dst}, nil
	}
}

func cmdList(layout Layout) error {
	models, err := scanModels(layout)
	if err != nil {
		return err
	}
	if len(models) == 0 {
		fmt.Println("No models found in models/.")
		return nil
	}
	for _, model := range models {
		kind := "text"
		if model.SupportsVision {
			kind = "vision"
		}
		fmt.Printf("- %s [%s] %s\n", model.Name, kind, model.LocalPath)
	}
	return nil
}

func cmdInfo(args []string, layout Layout) error {
	if len(args) < 1 {
		return errors.New("usage: myllm info <name>")
	}
	model, err := getModel(layout, args[0])
	if err != nil {
		return err
	}
	info, err := os.Stat(model.LocalPath)
	if err != nil {
		return err
	}
	fmt.Printf("Name: %s\nPath: %s\nSize: %.2f GB\nVision: %t\nMMProj: %s\n",
		model.Name,
		model.LocalPath,
		float64(info.Size())/1024/1024/1024,
		model.SupportsVision,
		model.MMProjPath,
	)
	return nil
}

func cmdRemove(args []string, layout Layout) error {
	parseArgs := args
	name := ""
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		name = args[0]
		parseArgs = args[1:]
	}
	fs := flag.NewFlagSet("rm", flag.ContinueOnError)
	deleteFiles := fs.Bool("delete-files", false, "also delete model files")
	if err := fs.Parse(parseArgs); err != nil {
		return err
	}
	if name == "" && fs.NArg() > 0 {
		name = fs.Arg(0)
	}
	if name == "" {
		return errors.New("usage: myllm rm <name> [--delete-files]")
	}

	model, err := getModel(layout, name)
	if err != nil {
		return err
	}
	if !*deleteFiles {
		fmt.Printf("Model %s exists at %s\nUse --delete-files to remove it.\n", model.Name, model.LocalPath)
		return nil
	}
	if err := os.Remove(model.LocalPath); err != nil {
		return err
	}
	if model.MMProjPath != "" {
		_ = os.Remove(model.MMProjPath)
	}
	fmt.Printf("Removed model %s\n", model.Name)
	return nil
}

func cmdDoctor(layout Layout) error {
	_, err := os.Stat(layout.BackendExePath)
	models, scanErr := scanModels(layout)
	if scanErr != nil {
		return scanErr
	}
	fmt.Printf("Platform: %s/%s\nLogical CPUs: %d\nPortable root: %s\nllama-server path: %s\nllama-server present: %t\nModels found: %d\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), layout.Root, layout.BackendExePath, err == nil, len(models))
	if err != nil {
		fmt.Println("Hint: put llama-server or llama-server.exe into backends/")
	}
	return nil
}

func cmdBench() error {
	threads := runtime.NumCPU() - 1
	if threads < 1 {
		threads = 1
	}
	fmt.Printf("Suggested CPU threads: %d\n", threads)
	fmt.Println("For 1B~4B Q4 models, start with context 4096 and temperature 0.7")
	return nil
}

func cmdChat(ctx context.Context, args []string, layout Layout, cfg Config) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	systemFile := fs.String("system-file", cfg.DefaultSystemPrompt, "system prompt text file")
	threads := fs.Int("threads", cfg.DefaultThreads, "CPU threads")
	contextLen := fs.Int("context", cfg.DefaultContextLength, "context length")
	temperature := fs.Float64("temp", cfg.DefaultTemperature, "temperature")
	topP := fs.Float64("top-p", cfg.DefaultTopP, "top-p")
	if err := fs.Parse(args); err != nil {
		return err
	}

	initialModel := cfg.DefaultModel
	if fs.NArg() > 0 {
		initialModel = fs.Arg(0)
	}
	if initialModel == "" {
		models, err := scanModels(layout)
		if err == nil && len(models) == 1 {
			initialModel = models[0].Name
		}
	}

	return runChatUI(ctx, layout, &cfg, ChatSession{
		ModelName:     initialModel,
		SystemFile:    *systemFile,
		Threads:       *threads,
		ContextLength: *contextLen,
		Temperature:   *temperature,
		TopP:          *topP,
	})
}

func cmdRun(ctx context.Context, args []string, layout Layout, cfg Config) error {
	modelName := ""
	parseArgs := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		modelName = args[0]
		parseArgs = args[1:]
	}

	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	var images multiFlag
	fs.Var(&images, "image", "image file path; repeatable")
	systemFile := fs.String("system-file", cfg.DefaultSystemPrompt, "system prompt text file")
	interactive := fs.Bool("interactive", false, "interactive chat loop")
	threads := fs.Int("threads", cfg.DefaultThreads, "CPU threads")
	contextLen := fs.Int("context", cfg.DefaultContextLength, "context length")
	temperature := fs.Float64("temp", cfg.DefaultTemperature, "temperature")
	topP := fs.Float64("top-p", cfg.DefaultTopP, "top-p")
	if err := fs.Parse(parseArgs); err != nil {
		return err
	}
	if modelName == "" && fs.NArg() > 0 {
		modelName = fs.Arg(0)
	}
	if modelName == "" {
		return errors.New("usage: myllm run <name> [prompt] [--image FILE] [--system-file FILE] [--interactive]")
	}

	model, err := getModel(layout, modelName)
	if err != nil {
		return err
	}

	remainingArgs := fs.Args()
	if len(remainingArgs) > 0 && remainingArgs[0] == modelName {
		remainingArgs = remainingArgs[1:]
	}
	prompt := strings.TrimSpace(strings.Join(remainingArgs, " "))
	if prompt == "" && !*interactive {
		if data, _ := readStdin(); strings.TrimSpace(data) != "" {
			prompt = strings.TrimSpace(data)
		}
	}
	if prompt == "" && !*interactive {
		return errors.New("missing prompt text")
	}

	loadedImages, err := loadImages([]string(images), cfg.MaxImageBytes)
	if err != nil {
		return err
	}
	if !filepath.IsAbs(*systemFile) {
		*systemFile = filepath.Join(layout.Root, *systemFile)
	}
	systemPrompt, err := loadPrompt(*systemFile)
	if err != nil {
		return err
	}

	history := []map[string]string{}
	if !*interactive {
		_, err := invokeModel(ctx, layout, cfg, model, history, systemPrompt, prompt, loadedImages, *threads, *contextLen, *temperature, *topP, func(token string) {
			fmt.Print(token)
		})
		fmt.Println()
		return err
	}

	if prompt != "" {
		answer, err := invokeModel(ctx, layout, cfg, model, history, systemPrompt, prompt, loadedImages, *threads, *contextLen, *temperature, *topP, func(token string) {
			fmt.Print(token)
		})
		fmt.Println()
		if err != nil {
			return err
		}
		history = append(history, map[string]string{"role": "user", "content": prompt}, map[string]string{"role": "assistant", "content": answer})
	}

	return runChatUI(ctx, layout, &cfg, ChatSession{
		ModelName:     model.Name,
		History:       history,
		SystemFile:    *systemFile,
		Threads:       *threads,
		ContextLength: *contextLen,
		Temperature:   *temperature,
		TopP:          *topP,
	})
}

func invokeModel(ctx context.Context, layout Layout, cfg Config, model LocalModel, history []map[string]string, systemPrompt, prompt string, images []Image, threads, ctxLen int, temperature, topP float64, onToken func(string)) (string, error) {
	baseURL, release, err := acquireServer(ctx, layout, cfg, model, threads, ctxLen)
	if err != nil {
		return "", err
	}
	defer release()

	timeout := time.Duration(cfg.RequestTimeoutSeconds) * time.Second
	activeSystemPrompt := buildEffectiveSystemPrompt(systemPrompt, cfg.EnableShellTool)
	activeHistory := append([]map[string]string(nil), history...)
	userPrompt := prompt
	activeImages := append([]Image(nil), images...)

	for step := 0; step < 6; step++ {
		suppressTokens := cfg.EnableShellTool && step == 0
		answer, err := runChatRequest(ctx, baseURL, model.Name, activeHistory, activeSystemPrompt, userPrompt, activeImages, temperature, topP, timeout, suppressTokens, onToken)
		if err != nil {
			if len(activeImages) > 0 && strings.Contains(err.Error(), "image_url") {
				answer, err = runLegacyChatRequest(ctx, baseURL, model.Name, activeHistory, activeSystemPrompt, userPrompt, activeImages, temperature, topP, timeout, suppressTokens, onToken)
			}
			if err != nil {
				return "", err
			}
		}
		if !cfg.EnableShellTool {
			return strings.TrimSpace(answer), nil
		}
		call, ok := parseShellToolCall(answer)
		if !ok {
			if onToken != nil && suppressTokens {
				onToken(answer)
			}
			return strings.TrimSpace(answer), nil
		}
		result, err := runShellTool(ctx, call)
		if err != nil {
			result = "shell tool error: " + err.Error()
		}
		activeHistory = append(activeHistory,
			map[string]string{"role": "user", "content": userPrompt},
			map[string]string{"role": "assistant", "content": answer},
			map[string]string{"role": "user", "content": "Shell tool result:\n" + result + "\nIf the task is complete, answer the user directly. Otherwise you may emit another <shell>...</shell> block."},
		)
		userPrompt = "Continue."
		activeImages = nil
	}
	return "", errors.New("tool loop exceeded maximum steps")
}

func acquireServer(ctx context.Context, layout Layout, cfg Config, model LocalModel, threads, ctxLen int) (string, func(), error) {
	if !cfg.KeepModelResident {
		port, err := freePort()
		if err != nil {
			return "", nil, err
		}
		cmd, baseURL, err := startServer(ctx, layout, cfg, model, threads, ctxLen, port)
		if err != nil {
			return "", nil, err
		}
		return baseURL, func() {
			stopServerProcess(cmd)
		}, nil
	}
	return sharedServer.acquire(ctx, layout, cfg, model, threads, ctxLen)
}

func startServer(ctx context.Context, layout Layout, cfg Config, model LocalModel, threads, ctxLen, port int) (*exec.Cmd, string, error) {
	if _, err := os.Stat(layout.BackendExePath); err != nil {
		return nil, "", fmt.Errorf("backend binary not found at %s", layout.BackendExePath)
	}

	args := []string{
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"-c", fmt.Sprintf("%d", ctxLen),
		"-t", fmt.Sprintf("%d", threads),
		"-m", model.LocalPath,
	}
	if cfg.EnableMMProj && model.MMProjPath != "" {
		args = append(args, "--mmproj", model.MMProjPath)
	}

	var startupOutput bytes.Buffer
	cmd := exec.CommandContext(ctx, layout.BackendExePath, args...)
	cmd.Stdout = &startupOutput
	cmd.Stderr = &startupOutput
	if err := cmd.Start(); err != nil {
		return nil, "", err
	}
	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()

	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	deadline := time.Now().Add(time.Duration(cfg.ServerStartupSeconds) * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-waitCh:
			return nil, "", explainServerStartFailure(err, startupOutput.String())
		default:
		}
		if ping(baseURL) == nil {
			return cmd, baseURL, nil
		}
		time.Sleep(700 * time.Millisecond)
	}

	_ = cmd.Process.Kill()
	err := <-waitCh
	if err == nil {
		return nil, "", errors.New("llama-server did not become ready in time")
	}
	return nil, "", fmt.Errorf("llama-server did not become ready in time: %s", summarizeServerOutput(startupOutput.String()))
}

func explainServerStartFailure(err error, output string) error {
	summary := summarizeServerOutput(output)
	code, ok := exitCodeOf(err)
	switch {
	case ok && runtime.GOOS == "windows" && code == -1073741515:
		return fmt.Errorf("llama-server exited before startup (exit code %d). This usually means missing DLL dependencies in backends/. Replace backends/ with the full official Windows package. Details: %s", code, summary)
	case ok:
		return fmt.Errorf("llama-server exited before startup (exit code %d). Details: %s", code, summary)
	case summary != "":
		return fmt.Errorf("llama-server exited before startup: %s", summary)
	default:
		return fmt.Errorf("llama-server exited before startup: %w", err)
	}
}

func (m *residentServerManager) acquire(ctx context.Context, layout Layout, cfg Config, model LocalModel, threads, ctxLen int) (string, func(), error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := residentServerKey(layout, model, threads, ctxLen)
	if m.idleTimer != nil {
		m.idleTimer.Stop()
		m.idleTimer = nil
	}
	if m.cmd != nil && m.key == key && ping(m.baseURL) == nil {
		m.inUse++
		return m.baseURL, func() { m.release(cfg) }, nil
	}
	if m.cmd != nil {
		m.stopLocked()
	}

	port, err := freePort()
	if err != nil {
		return "", nil, err
	}
	cmd, baseURL, err := startServer(ctx, layout, cfg, model, threads, ctxLen, port)
	if err != nil {
		return "", nil, err
	}
	m.cmd = cmd
	m.baseURL = baseURL
	m.key = key
	m.inUse = 1
	return m.baseURL, func() { m.release(cfg) }, nil
}

func (m *residentServerManager) release(cfg Config) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.inUse > 0 {
		m.inUse--
	}
	if m.inUse != 0 || m.cmd == nil {
		return
	}
	if cfg.ResidentIdleMinutes <= 0 {
		return
	}
	idle := time.Duration(cfg.ResidentIdleMinutes) * time.Minute
	m.idleTimer = time.AfterFunc(idle, func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		if m.inUse == 0 {
			m.stopLocked()
		}
	})
}

func (m *residentServerManager) shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopLocked()
}

func (m *residentServerManager) invalidate() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopLocked()
}

func (m *residentServerManager) stopLocked() {
	if m.idleTimer != nil {
		m.idleTimer.Stop()
		m.idleTimer = nil
	}
	stopServerProcess(m.cmd)
	m.cmd = nil
	m.baseURL = ""
	m.key = ""
	m.inUse = 0
}

func residentServerKey(layout Layout, model LocalModel, threads, ctxLen int) string {
	return strings.Join([]string{
		layout.BackendExePath,
		model.LocalPath,
		model.MMProjPath,
		strconv.Itoa(threads),
		strconv.Itoa(ctxLen),
	}, "|")
}

func stopServerProcess(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	_, _ = cmd.Process.Wait()
}

func summarizeServerOutput(output string) string {
	text := strings.TrimSpace(output)
	if text == "" {
		return "no stderr/stdout output"
	}
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	if len(lines) > 6 {
		lines = append(lines[:5], "...")
	}
	text = strings.Join(lines, " | ")
	if len(text) > 500 {
		text = text[:500] + "..."
	}
	return text
}

func exitCodeOf(err error) (int, bool) {
	if err == nil {
		return 0, false
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		return 0, false
	}
	return exitErr.ExitCode(), true
}

func runChatRequest(ctx context.Context, baseURL, model string, history []map[string]string, systemPrompt, prompt string, images []Image, temperature, topP float64, timeout time.Duration, suppressTokens bool, onToken func(string)) (string, error) {
	var builder strings.Builder
	collect := func(token string) {
		builder.WriteString(token)
		if !suppressTokens && onToken != nil {
			onToken(token)
		}
	}
	req := buildRequest(model, history, systemPrompt, prompt, images, temperature, topP)
	if err := streamChat(ctx, baseURL, req, timeout, collect); err != nil {
		return "", err
	}
	return strings.TrimSpace(builder.String()), nil
}

func runLegacyChatRequest(ctx context.Context, baseURL, model string, history []map[string]string, systemPrompt, prompt string, images []Image, temperature, topP float64, timeout time.Duration, suppressTokens bool, onToken func(string)) (string, error) {
	var builder strings.Builder
	collect := func(token string) {
		builder.WriteString(token)
		if !suppressTokens && onToken != nil {
			onToken(token)
		}
	}
	req := buildLegacyRequest(model, history, systemPrompt, prompt, images, temperature, topP)
	if err := streamChat(ctx, baseURL, req, timeout, collect); err != nil {
		return "", err
	}
	return strings.TrimSpace(builder.String()), nil
}

func buildEffectiveSystemPrompt(systemPrompt string, enableShellTool bool) string {
	if !enableShellTool {
		return systemPrompt
	}
	toolSpec := `

You can use one tool named shell.
When you need to run an operating system command, respond with ONLY:
<shell>
your command here
</shell>

Rules:
- Use the shell tool only when needed.
- Emit exactly one shell block and no extra prose when requesting tool execution.
- After receiving shell output, continue the task and answer normally unless another command is still necessary.
- Assume the environment is the user's local machine.
`
	return strings.TrimSpace(systemPrompt + "\n" + toolSpec)
}

func parseShellToolCall(answer string) (string, bool) {
	text := strings.TrimSpace(answer)
	start := strings.Index(text, "<shell>")
	end := strings.Index(text, "</shell>")
	if start < 0 || end < 0 || end <= start {
		return "", false
	}
	command := strings.TrimSpace(text[start+len("<shell>") : end])
	if command == "" {
		return "", false
	}
	return command, true
}

func runShellTool(ctx context.Context, command string) (string, error) {
	runCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(runCtx, "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command)
	} else {
		cmd = exec.CommandContext(runCtx, "sh", "-lc", command)
	}
	output, err := cmd.CombinedOutput()
	text := string(output)
	if len(text) > 32000 {
		text = text[:32000] + "\n...[truncated]"
	}
	if err != nil {
		return strings.TrimSpace(text), err
	}
	return strings.TrimSpace(text), nil
}

func ping(baseURL string) error {
	client := &http.Client{Timeout: 2 * time.Second}
	for _, ep := range []string{"/health", "/v1/models", "/"} {
		resp, err := client.Get(baseURL + ep)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 500 {
				return nil
			}
		}
	}
	return errors.New("not ready")
}

func buildRequest(model string, history []map[string]string, systemPrompt, prompt string, images []Image, temperature, topP float64) chatReq {
	messages := make([]map[string]any, 0, 2+len(history))
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, map[string]any{"role": "system", "content": systemPrompt})
	}
	for _, h := range history {
		messages = append(messages, map[string]any{"role": h["role"], "content": h["content"]})
	}
	if len(images) == 0 {
		messages = append(messages, map[string]any{"role": "user", "content": prompt})
	} else {
		content := []map[string]any{{"type": "text", "text": prompt}}
		for _, img := range images {
			content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": img.DataURL}})
		}
		messages = append(messages, map[string]any{"role": "user", "content": content})
	}
	return chatReq{Model: model, Messages: messages, Stream: true, Temperature: temperature, TopP: topP}
}

func buildLegacyRequest(model string, history []map[string]string, systemPrompt, prompt string, images []Image, temperature, topP float64) chatReq {
	imageData := make([]map[string]any, 0, len(images))
	for i, img := range images {
		if prompt != "" {
			prompt += "\n"
		}
		prompt += fmt.Sprintf("[img-%d]", i+1)
		imageData = append(imageData, map[string]any{"id": i + 1, "data": img.Base64})
	}
	messages := make([]map[string]any, 0, 2+len(history))
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, map[string]any{"role": "system", "content": systemPrompt})
	}
	for _, h := range history {
		messages = append(messages, map[string]any{"role": h["role"], "content": h["content"]})
	}
	messages = append(messages, map[string]any{"role": "user", "content": prompt})
	return chatReq{Model: model, Messages: messages, Stream: true, Temperature: temperature, TopP: topP, ImageData: imageData}
}

func streamChat(ctx context.Context, baseURL string, payload chatReq, timeout time.Duration, onToken func(string)) error {
	data, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 65536))
		return fmt.Errorf("llama-server error: %s", strings.TrimSpace(string(body)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 65536), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if raw == "[DONE]" {
			return nil
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			token := choice.Delta.Content
			if token == "" {
				token = choice.Message.Content
			}
			if token == "" {
				token = choice.Text
			}
			if token != "" {
				onToken(token)
			}
		}
	}
	return scanner.Err()
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func readStdin() (string, error) {
	st, err := os.Stdin.Stat()
	if err != nil {
		return "", err
	}
	if (st.Mode() & os.ModeCharDevice) != 0 {
		return "", nil
	}
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func loadPrompt(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func loadImages(paths []string, maxBytes int64) ([]Image, error) {
	out := make([]Image, 0, len(paths))
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return nil, err
		}
		if info.Size() > maxBytes {
			return nil, fmt.Errorf("image %s too large", p)
		}
		ext := strings.ToLower(filepath.Ext(p))
		mime := map[string]string{
			".jpg":  "image/jpeg",
			".jpeg": "image/jpeg",
			".png":  "image/png",
			".webp": "image/webp",
		}[ext]
		if mime == "" {
			return nil, fmt.Errorf("unsupported image format %s", ext)
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		enc := base64.StdEncoding.EncodeToString(data)
		out = append(out, Image{
			Path:      p,
			MIMEType:  mime,
			Base64:    enc,
			DataURL:   fmt.Sprintf("data:%s;base64,%s", mime, enc),
			SizeBytes: info.Size(),
		})
	}
	return out, nil
}

func copyModelInto(dstDir, src string) (string, error) {
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return "", err
	}
	srcAbs, err := filepath.Abs(src)
	if err != nil {
		return "", err
	}
	dst := filepath.Join(dstDir, filepath.Base(src))
	dstAbs, err := filepath.Abs(dst)
	if err != nil {
		return "", err
	}
	if strings.EqualFold(srcAbs, dstAbs) {
		return dstAbs, nil
	}
	data, err := os.ReadFile(srcAbs)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(dstAbs, data, 0o644); err != nil {
		return "", err
	}
	return dstAbs, nil
}

func downloadToFile(rawURL, dst, label string, progress func(DownloadProgress)) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	resp, err := http.Get(rawURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("download failed: %s", resp.Status)
	}
	total := resp.ContentLength

	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	start := time.Now()
	lastReport := time.Time{}
	buf := make([]byte, 1024*256)
	var downloaded int64
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := f.Write(buf[:n]); err != nil {
				_ = f.Close()
				_ = os.Remove(tmp)
				return err
			}
			downloaded += int64(n)
			now := time.Now()
			if progress != nil && (lastReport.IsZero() || now.Sub(lastReport) >= 120*time.Millisecond) {
				elapsed := now.Sub(start).Seconds()
				speed := 0.0
				if elapsed > 0 {
					speed = float64(downloaded) / elapsed
				}
				progress(DownloadProgress{Label: label, Downloaded: downloaded, Total: total, SpeedBytes: speed})
				lastReport = now
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			_ = f.Close()
			_ = os.Remove(tmp)
			return readErr
		}
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if progress != nil {
		elapsed := time.Since(start).Seconds()
		speed := 0.0
		if elapsed > 0 {
			speed = float64(downloaded) / elapsed
		}
		progress(DownloadProgress{Label: label, Downloaded: downloaded, Total: total, SpeedBytes: speed, Done: true})
	}
	return os.Rename(tmp, dst)
}

func downloadNamedModel(name, dst string, progress func(DownloadProgress)) error {
	if directURL, ok := resolveNamedModelDownload(name); ok {
		if err := downloadToFile(directURL, dst, "主模型", progress); err != nil {
			return err
		}
		if mmprojURL, ok := resolveNamedModelMMProjDownload(name); ok {
			mmprojDst := strings.TrimSuffix(dst, ".gguf") + ".mmproj.gguf"
			if err := downloadToFile(mmprojURL, mmprojDst, "mmproj", progress); err != nil {
				return err
			}
		}
		return nil
	}

	repo, tag := parseNamedModelReference(name)
	manifestURL := fmt.Sprintf("https://registry.ollama.ai/v2/%s/manifests/%s", repo, tag)

	req, err := http.NewRequest(http.MethodGet, manifestURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "myllm")
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 65536))
		return fmt.Errorf("named model lookup failed: %s %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var manifest ollamaManifest
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return err
	}

	modelDigest := ""
	for _, layer := range manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			modelDigest = layer.Digest
			break
		}
	}
	if modelDigest == "" {
		return errors.New("named model did not expose a downloadable model layer")
	}

	blobURL := fmt.Sprintf("https://registry.ollama.ai/v2/%s/blobs/%s", repo, modelDigest)
	return downloadToFile(blobURL, dst, name, progress)
}

func resolveNamedModelDownload(name string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "gemma4:e2b":
		return "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf?download=true", true
	default:
		return "", false
	}
}

func resolveNamedModelMMProjDownload(name string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "gemma4:e2b":
		return "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/mmproj-google_gemma-4-E2B-it-f16.gguf?download=true", true
	default:
		return "", false
	}
}

func parseNamedModelReference(name string) (string, string) {
	ref := strings.TrimSpace(name)
	repo := ref
	tag := "latest"
	if i := strings.LastIndex(ref, ":"); i >= 0 && i > strings.LastIndex(ref, "/") {
		repo = ref[:i]
		tag = ref[i+1:]
	}
	if !strings.Contains(repo, "/") {
		repo = "library/" + repo
	}
	return repo, tag
}

func looksLikeURL(value string) bool {
	u, err := url.Parse(strings.TrimSpace(value))
	return err == nil && (u.Scheme == "http" || u.Scheme == "https")
}

func guessName(rawURL string) string {
	name := filepath.Base(rawURL)
	if i := strings.Index(name, "?"); i >= 0 {
		name = name[:i]
	}
	if name == "." || name == "/" || name == "" {
		return "model.gguf"
	}
	return name
}

func encodeModelFilename(name string) string {
	encoded := url.QueryEscape(strings.TrimSpace(name))
	encoded = strings.ReplaceAll(encoded, "+", "%20")
	return encoded
}

func decodeModelName(base string) string {
	if value, err := url.QueryUnescape(strings.ReplaceAll(base, "%20", "+")); err == nil && value != "" {
		return value
	}
	return base
}

func formatDownloadProgress(p DownloadProgress) string {
	progressBar := renderProgressBar(p.Downloaded, p.Total, 28)
	total := "unknown"
	if p.Total > 0 {
		total = humanBytes(p.Total)
	}
	return fmt.Sprintf("%s %s/%s %s/s %s", progressBar, humanBytes(p.Downloaded), total, humanBytes(int64(p.SpeedBytes)), p.Label)
}

func renderProgressBar(downloaded, total int64, width int) string {
	if width < 10 {
		width = 10
	}
	if total <= 0 {
		return "[" + strings.Repeat("=", width/3) + strings.Repeat(" ", width-(width/3)) + "]"
	}
	ratio := float64(downloaded) / float64(total)
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	filled := int(ratio * float64(width))
	if filled > width {
		filled = width
	}
	return fmt.Sprintf("[%s%s] %3.0f%%", strings.Repeat("=", filled), strings.Repeat(" ", width-filled), ratio*100)
}

func humanBytes(value int64) string {
	if value < 1024 {
		return fmt.Sprintf("%d B", value)
	}
	units := []string{"KB", "MB", "GB", "TB"}
	size := float64(value)
	unit := ""
	for _, u := range units {
		size /= 1024
		unit = u
		if size < 1024 {
			break
		}
	}
	if size >= 100 {
		return fmt.Sprintf("%.0f %s", size, unit)
	}
	if size >= 10 {
		return fmt.Sprintf("%.1f %s", size, unit)
	}
	return fmt.Sprintf("%.2f %s", size, unit)
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

func parsePositiveInt(raw, field string) (int, error) {
	if raw == "" {
		return 0, fmt.Errorf("missing value for %s", field)
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("invalid %s value %q", field, raw)
	}
	return v, nil
}

func parseFloat(raw, field string) (float64, error) {
	if raw == "" {
		return 0, fmt.Errorf("missing value for %s", field)
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s value %q", field, raw)
	}
	return v, nil
}
