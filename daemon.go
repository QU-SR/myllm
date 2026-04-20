package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type daemonChatRequest struct {
	ModelName     string              `json:"model"`
	History       []map[string]string `json:"history"`
	SystemPrompt  string              `json:"system_prompt"`
	Prompt        string              `json:"prompt"`
	Images        []Image             `json:"images,omitempty"`
	Threads       int                 `json:"threads"`
	ContextLength int                 `json:"context_length"`
	Temperature   float64             `json:"temperature"`
	TopP          float64             `json:"top_p"`
}

type daemonTokenEvent struct {
	Token string `json:"token,omitempty"`
	Error string `json:"error,omitempty"`
}

func daemonBaseURL(cfg Config) string {
	port := cfg.DaemonPort
	if port == 0 {
		port = defaultConfig().DaemonPort
	}
	return fmt.Sprintf("http://127.0.0.1:%d", port)
}

func ensureDaemon(ctx context.Context, layout Layout, cfg Config) (string, error) {
	baseURL := daemonBaseURL(cfg)
	if pingDaemon(baseURL) == nil {
		return baseURL, nil
	}
	if err := startDaemonProcess(layout); err != nil {
		return "", err
	}
	deadline := time.Now().Add(12 * time.Second)
	for time.Now().Before(deadline) {
		if pingDaemon(baseURL) == nil {
			return baseURL, nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
	return "", errors.New("background daemon did not become ready")
}

func startDaemonProcess(layout Layout) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(exe, "daemon")
	cmd.Dir = layout.Root
	cmd.Env = append(os.Environ(), "PORTABLELLM_HOME="+layout.Root)
	configureDetachedProcess(cmd)
	return cmd.Start()
}

func pingDaemon(baseURL string) error {
	client := &http.Client{Timeout: 800 * time.Millisecond}
	resp, err := client.Get(baseURL + "/healthz")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("daemon health returned %s", resp.Status)
	}
	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return err
	}
	if body.Name != "myllm" {
		return errors.New("unexpected daemon response")
	}
	return nil
}

func cmdDaemon(ctx context.Context, layout Layout, cfg Config) error {
	addr := fmt.Sprintf("127.0.0.1:%d", cfg.DaemonPort)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"name":       "myllm",
			"status":     "ok",
			"api_base":   "/v1",
			"daemon_url": daemonBaseURL(cfg),
		})
	})
	mux.HandleFunc("/myllm/chat", func(w http.ResponseWriter, r *http.Request) {
		handleDaemonChat(ctx, w, r, layout, cfg)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		handleOpenAIModels(w, layout)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		handleOpenAIChatCompletions(ctx, w, r, layout, cfg)
	})

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	server := &http.Server{Addr: addr, Handler: mux}
	serverErr := make(chan error, 1)
	go func() {
		err := server.Serve(listener)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
			return
		}
		serverErr <- nil
	}()

	trayErrCh := make(chan error, 1)
	go func() {
		trayErrCh <- runTray(layout, daemonBaseURL(cfg))
	}()

	var trayErr error
	select {
	case err := <-serverErr:
		requestTrayQuit()
		trayErr = <-trayErrCh
		if err != nil {
			if trayErr != nil {
				return fmt.Errorf("%w; tray: %v", err, trayErr)
			}
			return err
		}
	case trayErr = <-trayErrCh:
	case <-ctx.Done():
		requestTrayQuit()
		trayErr = <-trayErrCh
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdownCtx)
	sharedServer.shutdown()

	if trayErr != nil {
		return trayErr
	}
	return nil
}

func handleDaemonChat(ctx context.Context, w http.ResponseWriter, r *http.Request, layout Layout, cfg Config) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	cfg = reloadDaemonConfig(layout, cfg)
	var req daemonChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 32*1024*1024)).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	modelName := req.ModelName
	if modelName == "" {
		modelName = cfg.DefaultModel
	}
	model, err := resolveRequestedModel(layout, modelName, cfg)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Threads <= 0 {
		req.Threads = cfg.DefaultThreads
	}
	if req.ContextLength <= 0 {
		req.ContextLength = cfg.DefaultContextLength
	}
	if req.Temperature == 0 {
		req.Temperature = cfg.DefaultTemperature
	}
	if req.TopP == 0 {
		req.TopP = cfg.DefaultTopP
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	_, err = invokeModel(ctx, layout, cfg, model, req.History, req.SystemPrompt, req.Prompt, req.Images, req.Threads, req.ContextLength, req.Temperature, req.TopP, func(token string) {
		writeSSE(w, daemonTokenEvent{Token: token})
		flush()
	})
	if err != nil {
		writeSSE(w, daemonTokenEvent{Error: err.Error()})
		flush()
		return
	}
	fmt.Fprint(w, "event: done\ndata: {}\n\n")
	flush()
}

func handleOpenAIModels(w http.ResponseWriter, layout Layout) {
	setAPIHeaders(w)
	models, err := scanModels(layout)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	data := make([]map[string]any, 0, len(models))
	for _, model := range models {
		data = append(data, map[string]any{
			"id":       model.Name,
			"object":   "model",
			"owned_by": "myllm",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
	})
}

func handleOpenAIChatCompletions(ctx context.Context, w http.ResponseWriter, r *http.Request, layout Layout, cfg Config) {
	setAPIHeaders(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	cfg = reloadDaemonConfig(layout, cfg)
	body, err := io.ReadAll(io.LimitReader(r.Body, 32*1024*1024))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var req chatReq
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	model, err := resolveRequestedModel(layout, req.Model, cfg)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	req.Model = model.Name

	baseURL, release, err := acquireServer(ctx, layout, cfg, model, cfg.DefaultThreads, cfg.DefaultContextLength)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer release()

	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(mustMarshal(req)))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: time.Duration(cfg.RequestTimeoutSeconds) * time.Second}).Do(proxyReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(resp.StatusCode)
	if req.Stream {
		flusher, _ := w.(http.Flusher)
		reader := bufio.NewReader(resp.Body)
		buf := make([]byte, 32*1024)
		for {
			n, readErr := reader.Read(buf)
			if n > 0 {
				_, _ = w.Write(buf[:n])
				if flusher != nil {
					flusher.Flush()
				}
			}
			if readErr != nil {
				break
			}
		}
		return
	}
	_, _ = io.Copy(w, resp.Body)
}

func reloadDaemonConfig(layout Layout, fallback Config) Config {
	cfg, err := loadConfig(layout)
	if err != nil {
		return fallback
	}
	if cfg.DaemonPort == 0 {
		cfg.DaemonPort = fallback.DaemonPort
	}
	return cfg
}

func resolveRequestedModel(layout Layout, requested string, cfg Config) (LocalModel, error) {
	if strings.TrimSpace(requested) != "" {
		if model, err := getModel(layout, requested); err == nil {
			return model, nil
		}
	}
	if strings.TrimSpace(cfg.DefaultModel) != "" {
		if model, err := getModel(layout, cfg.DefaultModel); err == nil {
			return model, nil
		}
	}
	models, err := scanModels(layout)
	if err != nil {
		return LocalModel{}, err
	}
	if len(models) == 1 {
		return models[0], nil
	}
	if requested == "" {
		return LocalModel{}, errors.New("missing model; set default_model or pass an installed model name")
	}
	return LocalModel{}, fmt.Errorf("model %q not found", requested)
}

func invokeDaemonModel(ctx context.Context, baseURL string, req daemonChatRequest, timeout time.Duration, onToken func(string)) (string, error) {
	data, _ := json.Marshal(req)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/myllm/chat", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: timeout}).Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 65536))
		return "", fmt.Errorf("daemon error: %s", strings.TrimSpace(string(body)))
	}
	var answer strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 65536), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "event:") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if raw == "{}" {
			break
		}
		var event daemonTokenEvent
		if err := json.Unmarshal([]byte(raw), &event); err != nil {
			continue
		}
		if event.Error != "" {
			return strings.TrimSpace(answer.String()), errors.New(event.Error)
		}
		if event.Token != "" {
			answer.WriteString(event.Token)
			if onToken != nil {
				onToken(event.Token)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return strings.TrimSpace(answer.String()), err
	}
	return strings.TrimSpace(answer.String()), nil
}

func writeSSE(w io.Writer, event daemonTokenEvent) {
	data, _ := json.Marshal(event)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func setAPIHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}

func mustMarshal(value any) []byte {
	data, _ := json.Marshal(value)
	return data
}

func launchChatWindow(layout Layout) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command("cmd", "/c", "start", "", exe, "chat")
	cmd.Dir = layout.Root
	cmd.Env = append(os.Environ(), "PORTABLELLM_HOME="+layout.Root)
	configureLauncherProcess(cmd)
	if err := cmd.Start(); err != nil {
		_ = appendLog(layout, "tray-open.log", "failed to open chat window: "+err.Error())
		return err
	}
	_ = appendLog(layout, "tray-open.log", "chat window launch requested")
	return nil
}

func appendLog(layout Layout, name, text string) error {
	logDir := filepath.Join(layout.Root, "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return err
	}
	line := time.Now().Format(time.RFC3339) + " " + text + "\n"
	f, err := os.OpenFile(filepath.Join(logDir, name), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line)
	return err
}
