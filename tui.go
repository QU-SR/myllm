package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

type uiMode int

const (
	modeChat uiMode = iota
	modePromptPull
	modePromptSystem
	modePromptThreads
	modePromptContext
	modePromptTemp
	modePromptTopP
	modePromptShell
	modeSelectModel
)

type performanceDraft struct {
	Profile       string
	Threads       int
	ContextLength int
	Resident      bool
	IdleMinutes   int
	EnableMMProj  bool
}

type pendingAttachment struct {
	Kind        string
	Path        string
	Label       string
	FileContent string
}

type uiMessage struct {
	Role    string
	Content string
}

type slashCommand struct {
	Name        string
	Description string
}

type streamStartMsg struct {
	stream <-chan tea.Msg
}

type streamTokenMsg struct {
	Text string
}

type streamDoneMsg struct {
	Answer string
	Err    error
}

type pullDoneMsg struct {
	ModelName string
	Dest      string
	Err       error
}

type pullStartMsg struct {
	stream <-chan tea.Msg
}

type pullProgressMsg struct {
	Progress DownloadProgress
}

type chatUI struct {
	ctx       context.Context
	layout    Layout
	cfg       *Config
	daemonURL string

	mode uiMode

	input textinput.Model

	width  int
	height int

	modelName string
	history   []map[string]string
	messages  []uiMessage

	attachments []pendingAttachment
	busy        bool

	stream     <-chan tea.Msg
	pullStream <-chan tea.Msg

	downloadProgress *DownloadProgress
	lastPullStage    string

	slashIndex int

	selectTitle        string
	selectItems        []string
	selectDescriptions []string
	selectIndex        int
	selectFlow         string

	systemFile        string
	threads           int
	contextLength     int
	temperature       float64
	topP              float64
	pendingUserPrompt string
	perfDraft         performanceDraft
}

func runChatUI(ctx context.Context, layout Layout, cfg *Config, session ChatSession) error {
	input := textinput.New()
	input.Focus()
	input.Prompt = "> "
	input.Placeholder = ""
	input.CharLimit = 0

	if session.SystemFile == "" {
		session.SystemFile = cfg.DefaultSystemPrompt
	}
	if !filepath.IsAbs(session.SystemFile) {
		session.SystemFile = filepath.Join(layout.Root, session.SystemFile)
	}

	ui := chatUI{
		ctx:           ctx,
		layout:        layout,
		cfg:           cfg,
		daemonURL:     session.DaemonURL,
		mode:          modeChat,
		input:         input,
		modelName:     session.ModelName,
		history:       append([]map[string]string(nil), session.History...),
		messages:      historyToMessages(session.History),
		systemFile:    session.SystemFile,
		threads:       session.Threads,
		contextLength: session.ContextLength,
		temperature:   session.Temperature,
		topP:          session.TopP,
	}
	if ui.modelName == "" {
		models, err := scanModels(layout)
		if err == nil && len(models) == 1 {
			ui.modelName = models[0].Name
		}
	}

	program := tea.NewProgram(ui, tea.WithAltScreen())
	_, err := program.Run()
	return err
}

func historyToMessages(history []map[string]string) []uiMessage {
	out := make([]uiMessage, 0, len(history))
	for _, item := range history {
		out = append(out, uiMessage{
			Role:    item["role"],
			Content: item["content"],
		})
	}
	return out
}

func (m chatUI) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, tea.ClearScreen)
}

func (m chatUI) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.Width = max(20, msg.Width-4)
		return m, tea.ClearScreen

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		}

		if msg.Paste {
			if m.handlePaste(string(msg.Runes)) {
				return m, nil
			}
		}

		switch m.mode {
		case modeSelectModel:
			return m.updateSelectMode(msg)
		case modePromptPull, modePromptSystem, modePromptThreads, modePromptContext, modePromptTemp, modePromptTopP, modePromptShell:
			return m.updatePromptMode(msg)
		default:
			return m.updateChatMode(msg)
		}

	case streamStartMsg:
		m.stream = msg.stream
		return m, waitForStream(m.stream)

	case pullStartMsg:
		m.pullStream = msg.stream
		return m, waitForPull(m.pullStream)

	case streamTokenMsg:
		if len(m.messages) == 0 || m.messages[len(m.messages)-1].Role != "assistant" {
			m.messages = append(m.messages, uiMessage{Role: "assistant"})
		}
		m.messages[len(m.messages)-1].Content += msg.Text
		return m, waitForStream(m.stream)

	case streamDoneMsg:
		m.busy = false
		if msg.Err != nil {
			if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" && m.messages[len(m.messages)-1].Content == "" {
				m.messages[len(m.messages)-1].Content = "[错误] " + msg.Err.Error()
			}
			m.appendSystemMessage("Error: " + msg.Err.Error())
			return m, nil
		}
		m.history = append(m.history,
			map[string]string{"role": "user", "content": m.pendingUserPrompt},
			map[string]string{"role": "assistant", "content": msg.Answer},
		)
		m.pendingUserPrompt = ""
		return m, nil

	case pullProgressMsg:
		progress := msg.Progress
		stage := strings.TrimSpace(progress.Label)
		if stage != "" && stage != m.lastPullStage {
			switch stage {
			case "主模型":
				m.appendSystemMessage("正在下载主模型...")
			case "mmproj":
				m.appendSystemMessage("正在下载 mmproj...")
			}
			m.lastPullStage = stage
		}
		m.downloadProgress = &progress
		return m, waitForPull(m.pullStream)

	case pullDoneMsg:
		m.busy = false
		m.downloadProgress = nil
		m.lastPullStage = ""
		if msg.Err != nil {
			m.appendSystemMessage("Download failed: " + msg.Err.Error())
			return m, nil
		}
		m.modelName = msg.ModelName
		m.cfg.DefaultModel = msg.ModelName
		_ = saveConfig(m.layout, *m.cfg)
		m.appendSystemMessage("Model downloaded and selected: " + msg.ModelName)
		return m, nil
	}

	return m, nil
}

func (m chatUI) updateChatMode(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.busy {
		return m, nil
	}

	if m.isSlashMode() {
		switch msg.Type {
		case tea.KeyUp:
			commands := m.filteredCommands()
			if len(commands) > 0 {
				m.slashIndex = (m.slashIndex - 1 + len(commands)) % len(commands)
			}
			return m, nil
		case tea.KeyDown:
			commands := m.filteredCommands()
			if len(commands) > 0 {
				m.slashIndex = (m.slashIndex + 1) % len(commands)
			}
			return m, nil
		case tea.KeyEnter:
			commands := m.filteredCommands()
			if len(commands) == 0 {
				m.appendSystemMessage("No matching command.")
				return m, nil
			}
			return m.executeSlashCommand(commands[m.slashIndex].Name)
		}
	}

	switch msg.Type {
	case tea.KeyEnter:
		return m.sendCurrentMessage()
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	if m.isSlashMode() {
		commands := m.filteredCommands()
		if m.slashIndex >= len(commands) {
			m.slashIndex = 0
		}
	}
	return m, cmd
}

func (m chatUI) updatePromptMode(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEsc:
		m.mode = modeChat
		m.input.Reset()
		m.input.Placeholder = ""
		return m, nil
	case tea.KeyEnter:
		value := strings.TrimSpace(m.input.Value())
		activeMode := m.mode
		m.input.Reset()
		m.mode = modeChat
		m.input.Placeholder = ""
		return m.handlePromptSubmit(activeMode, value)
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m chatUI) updateSelectMode(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEsc:
		switch m.selectFlow {
		case "perf-resident":
			return m.openPerformanceProfileMenu(), nil
		case "perf-idle":
			return m.openPerformanceResidentMenu(), nil
		case "perf-mmproj":
			if m.perfDraft.Resident {
				return m.openPerformanceIdleMenu(), nil
			}
			return m.openPerformanceResidentMenu(), nil
		default:
			m.mode = modeChat
			m.selectTitle = ""
			m.selectItems = nil
			m.selectDescriptions = nil
			m.selectFlow = ""
		}
		return m, nil
	case tea.KeyUp:
		if len(m.selectItems) > 0 {
			m.selectIndex = (m.selectIndex - 1 + len(m.selectItems)) % len(m.selectItems)
		}
		return m, nil
	case tea.KeyDown:
		if len(m.selectItems) > 0 {
			m.selectIndex = (m.selectIndex + 1) % len(m.selectItems)
		}
		return m, nil
	case tea.KeyEnter:
		if len(m.selectItems) == 0 {
			m.mode = modeChat
			return m, nil
		}
		selected := m.selectItems[m.selectIndex]
		switch m.selectFlow {
		case "model":
			m.modelName = selected
			m.cfg.DefaultModel = selected
			_ = saveConfig(m.layout, *m.cfg)
			m.mode = modeChat
			m.selectTitle = ""
			m.selectItems = nil
			m.selectDescriptions = nil
			m.selectFlow = ""
			m.appendSystemMessage("Model selected: " + selected)
			return m, nil
		case "perf-profile":
			switch selected {
			case "极速响应":
				m.perfDraft.Profile = "speed"
				m.perfDraft.Threads = speedProfileThreads()
				m.perfDraft.ContextLength = 4096
			case "省内存":
				m.perfDraft.Profile = "memory"
				m.perfDraft.Threads = memoryProfileThreads()
				m.perfDraft.ContextLength = 2048
			default:
				m.perfDraft.Profile = "balanced"
				m.perfDraft.Threads = balancedProfileThreads()
				m.perfDraft.ContextLength = 4096
			}
			return m.openPerformanceResidentMenu(), nil
		case "perf-resident":
			m.perfDraft.Resident = selected == "常驻当前模型"
			if m.perfDraft.Resident {
				return m.openPerformanceIdleMenu(), nil
			}
			m.perfDraft.IdleMinutes = 0
			return m.openPerformanceMMProjMenu(), nil
		case "perf-idle":
			switch selected {
			case "5 分钟后释放":
				m.perfDraft.IdleMinutes = 5
			case "15 分钟后释放":
				m.perfDraft.IdleMinutes = 15
			case "30 分钟后释放":
				m.perfDraft.IdleMinutes = 30
			case "60 分钟后释放":
				m.perfDraft.IdleMinutes = 60
			default:
				m.perfDraft.IdleMinutes = 0
			}
			return m.openPerformanceMMProjMenu(), nil
		case "perf-mmproj":
			m.perfDraft.EnableMMProj = m.selectIndex == 0
			return m.applyPerformanceDraft(), nil
		default:
			m.mode = modeChat
			return m, nil
		}
	}
	return m, nil
}

func (m chatUI) handlePromptSubmit(mode uiMode, value string) (tea.Model, tea.Cmd) {
	switch mode {
	case modePromptPull:
		if value == "" {
			return m, nil
		}
		m.busy = true
		m.appendSystemMessage("Downloading model...")
		return m, pullModelCmd(m.layout, value)

	case modePromptSystem:
		if value == "" {
			return m, nil
		}
		path := value
		if !filepath.IsAbs(path) {
			path = filepath.Join(m.layout.Root, path)
		}
		if _, err := os.Stat(path); err != nil {
			m.appendSystemMessage("System prompt file not found.")
			return m, nil
		}
		m.systemFile = path
		m.cfg.DefaultSystemPrompt = path
		_ = saveConfig(m.layout, *m.cfg)
		m.appendSystemMessage("System prompt updated.")
		return m, nil

	case modePromptThreads:
		v, err := parsePositiveInt(value, "threads")
		if err != nil {
			m.appendSystemMessage(err.Error())
			return m, nil
		}
		m.threads = v
		m.cfg.DefaultThreads = v
		_ = saveConfig(m.layout, *m.cfg)
		m.appendSystemMessage(fmt.Sprintf("Threads set to %d.", v))
		return m, nil

	case modePromptContext:
		v, err := parsePositiveInt(value, "context")
		if err != nil {
			m.appendSystemMessage(err.Error())
			return m, nil
		}
		m.contextLength = v
		m.cfg.DefaultContextLength = v
		_ = saveConfig(m.layout, *m.cfg)
		m.appendSystemMessage(fmt.Sprintf("Context length set to %d.", v))
		return m, nil

	case modePromptTemp:
		v, err := parseFloat(value, "temp")
		if err != nil {
			m.appendSystemMessage(err.Error())
			return m, nil
		}
		m.temperature = v
		m.cfg.DefaultTemperature = v
		_ = saveConfig(m.layout, *m.cfg)
		m.appendSystemMessage(fmt.Sprintf("Temperature set to %.2f.", v))
		return m, nil

	case modePromptTopP:
		v, err := parseFloat(value, "top-p")
		if err != nil {
			m.appendSystemMessage(err.Error())
			return m, nil
		}
		m.topP = v
		m.cfg.DefaultTopP = v
		_ = saveConfig(m.layout, *m.cfg)
		m.appendSystemMessage(fmt.Sprintf("Top-p set to %.2f.", v))
		return m, nil

	case modePromptShell:
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "on", "enable", "enabled", "true", "1":
			m.cfg.EnableShellTool = true
			_ = saveConfig(m.layout, *m.cfg)
			m.appendSystemMessage("Shell tool enabled.")
		case "off", "disable", "disabled", "false", "0":
			m.cfg.EnableShellTool = false
			_ = saveConfig(m.layout, *m.cfg)
			m.appendSystemMessage("Shell tool disabled.")
		case "":
			return m, nil
		default:
			m.appendSystemMessage("Enter on or off.")
		}
		return m, nil
	}

	return m, nil
}

func (m chatUI) executeSlashCommand(name string) (tea.Model, tea.Cmd) {
	m.input.Reset()
	m.slashIndex = 0

	switch name {
	case "help":
		m.appendSystemMessage("Commands: /help /model /pull /performance /shell /clear /system /threads /context /temp /top-p /stats /doctor /exit")
		return m, nil
	case "model":
		models, err := scanModels(m.layout)
		if err != nil {
			m.appendSystemMessage("Failed to read models: " + err.Error())
			return m, nil
		}
		if len(models) == 0 {
			m.appendSystemMessage("No models found in models/. Use /pull first.")
			return m, nil
		}
		m.mode = modeSelectModel
		m.selectTitle = "Select model"
		m.selectItems = make([]string, 0, len(models))
		m.selectDescriptions = nil
		m.selectFlow = "model"
		for _, model := range models {
			m.selectItems = append(m.selectItems, model.Name)
		}
		m.selectIndex = 0
		m.appendSystemMessage("Select a model.")
		return m, nil
	case "pull":
		m.mode = modePromptPull
		m.input.Placeholder = ""
		m.appendSystemMessage("Enter a model name or GGUF direct URL, then press Enter.")
		return m, nil
	case "performance":
		return m.openPerformanceProfileMenu(), nil
	case "shell":
		m.mode = modePromptShell
		m.input.Placeholder = ""
		if m.cfg.EnableShellTool {
			m.appendSystemMessage("Shell tool is ON. Enter on/off to change it.")
		} else {
			m.appendSystemMessage("Shell tool is OFF. Enter on/off to change it.")
		}
		return m, nil
	case "clear":
		m.history = nil
		m.messages = nil
		m.attachments = nil
		return m, tea.ClearScreen
	case "system":
		m.mode = modePromptSystem
		m.input.Placeholder = ""
		m.appendSystemMessage("Enter a system prompt file path.")
		return m, nil
	case "threads":
		m.mode = modePromptThreads
		m.input.Placeholder = ""
		m.appendSystemMessage(fmt.Sprintf("Current threads: %d. Enter a new value.", m.threads))
		return m, nil
	case "context":
		m.mode = modePromptContext
		m.input.Placeholder = ""
		m.appendSystemMessage(fmt.Sprintf("Current context length: %d. Enter a new value.", m.contextLength))
		return m, nil
	case "temp":
		m.mode = modePromptTemp
		m.input.Placeholder = ""
		m.appendSystemMessage(fmt.Sprintf("Current temperature: %.2f. Enter a new value.", m.temperature))
		return m, nil
	case "top-p":
		m.mode = modePromptTopP
		m.input.Placeholder = ""
		m.appendSystemMessage(fmt.Sprintf("Current top-p: %.2f. Enter a new value.", m.topP))
		return m, nil
	case "stats":
		resident := "off"
		if m.cfg.KeepModelResident {
			if m.cfg.ResidentIdleMinutes > 0 {
				resident = fmt.Sprintf("on/%dmin", m.cfg.ResidentIdleMinutes)
			} else {
				resident = "on/always"
			}
		}
		mmproj := "off"
		if m.cfg.EnableMMProj {
			mmproj = "on"
		}
		m.appendSystemMessage(fmt.Sprintf("Model=%s Threads=%d Context=%d Temp=%.2f Top-p=%.2f Shell=%t Resident=%s MMProj=%s Profile=%s Attachments=%d",
			emptyFallback(m.modelName, "none"),
			m.threads,
			m.contextLength,
			m.temperature,
			m.topP,
			m.cfg.EnableShellTool,
			resident,
			mmproj,
			emptyFallback(m.cfg.PerformanceProfile, "balanced"),
			len(m.attachments),
		))
		return m, nil
	case "doctor":
		backend := "缺失"
		if _, err := os.Stat(m.layout.BackendExePath); err == nil {
			backend = "ready"
		}
		models, _ := scanModels(m.layout)
		m.appendSystemMessage(fmt.Sprintf("Root=%s Backend=%s Models=%d", m.layout.Root, backend, len(models)))
		return m, nil
	case "exit":
		return m, tea.Quit
	default:
		m.appendSystemMessage("Unknown command.")
		return m, nil
	}
}

func (m chatUI) sendCurrentMessage() (tea.Model, tea.Cmd) {
	text := strings.TrimSpace(m.input.Value())
	if paths, remainder, ok := parseDroppedPaths(text); ok {
		for _, path := range paths {
			item, err := attachmentFromPath(path, m.cfg.MaxFileBytes, m.countAttachmentsByKind("image"), m.countAttachmentsByKind("file"))
			if err != nil {
				m.appendSystemMessage(err.Error())
				continue
			}
			m.attachments = append(m.attachments, item)
		}
		text = strings.TrimSpace(remainder)
		if m.input.Value() != text {
			m.input.SetValue(text)
		}
	}
	if text == "" && len(m.attachments) == 0 {
		return m, nil
	}
	if m.modelName == "" {
		m.appendSystemMessage("Select a model first with /model or /pull.")
		return m, nil
	}

	model, err := getModel(m.layout, m.modelName)
	if err != nil {
		m.appendSystemMessage("Current model is unavailable: " + err.Error())
		return m, nil
	}
	if m.countAttachmentsByKind("image") > 0 {
		switch {
		case !m.cfg.EnableMMProj:
			m.appendSystemMessage("当前已关闭 mmproj，图片输入不可用。请到 /performance 中开启后再试。")
			return m, nil
		case model.MMProjPath == "":
			m.appendSystemMessage("此模型不支持图片，请切换支持视觉的模型")
			return m, nil
		}
	}
	systemPrompt, err := loadPrompt(m.systemFile)
	if err != nil {
		m.appendSystemMessage("Failed to read system prompt: " + err.Error())
		return m, nil
	}

	prompt, imagePaths, displayText := buildUserPrompt(text, m.attachments)
	images, err := loadImages(imagePaths, m.cfg.MaxImageBytes)
	if err != nil {
		m.appendSystemMessage("Failed to load image: " + err.Error())
		return m, nil
	}

	m.messages = append(m.messages, uiMessage{Role: "user", Content: displayText})
	m.messages = append(m.messages, uiMessage{Role: "assistant", Content: ""})
	m.pendingUserPrompt = prompt
	m.attachments = nil
	m.input.Reset()
	m.busy = true

	return m, streamAnswerCmd(m.ctx, m.daemonURL, m.layout, *m.cfg, model, append([]map[string]string(nil), m.history...), systemPrompt, prompt, images, m.threads, m.contextLength, m.temperature, m.topP)
}

func buildUserPrompt(text string, attachments []pendingAttachment) (prompt string, imagePaths []string, display string) {
	labels := make([]string, 0, len(attachments))
	fileBlocks := make([]string, 0)
	for _, item := range attachments {
		labels = append(labels, item.Label)
		if item.Kind == "image" {
			imagePaths = append(imagePaths, item.Path)
			continue
		}
		fileBlocks = append(fileBlocks, fmt.Sprintf("File: %s\n%s", filepath.Base(item.Path), item.FileContent))
	}

	display = strings.TrimSpace(strings.Join([]string{strings.Join(labels, " "), text}, " "))
	prompt = strings.TrimSpace(text)
	if len(fileBlocks) > 0 {
		if prompt != "" {
			prompt += "\n\n"
		}
		prompt += "Attached file contents:\n" + strings.Join(fileBlocks, "\n\n")
	}
	if prompt == "" && len(labels) > 0 {
		prompt = strings.Join(labels, " ")
	}
	return prompt, imagePaths, display
}

func streamAnswerCmd(ctx context.Context, daemonURL string, layout Layout, cfg Config, model LocalModel, history []map[string]string, systemPrompt, prompt string, images []Image, threads, contextLength int, temperature, topP float64) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg)
		go func() {
			var answer string
			var err error
			onToken := func(token string) {
				ch <- streamTokenMsg{Text: token}
			}
			if strings.TrimSpace(daemonURL) != "" {
				answer, err = invokeDaemonModel(ctx, daemonURL, daemonChatRequest{
					ModelName:     model.Name,
					History:       history,
					SystemPrompt:  systemPrompt,
					Prompt:        prompt,
					Images:        images,
					Threads:       threads,
					ContextLength: contextLength,
					Temperature:   temperature,
					TopP:          topP,
				}, time.Duration(cfg.RequestTimeoutSeconds)*time.Second, onToken)
			} else {
				answer, err = invokeModel(ctx, layout, cfg, model, history, systemPrompt, prompt, images, threads, contextLength, temperature, topP, onToken)
			}
			ch <- streamDoneMsg{Answer: answer, Err: err}
			close(ch)
		}()
		return streamStartMsg{stream: ch}
	}
}

func waitForStream(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return nil
		}
		return msg
	}
}

func pullModelCmd(layout Layout, value string) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg)
		go func() {
			result, err := pullModel([]string{value}, layout, func(progress DownloadProgress) {
				ch <- pullProgressMsg{Progress: progress}
			})
			ch <- pullDoneMsg{ModelName: result.ModelName, Dest: result.Dest, Err: err}
			close(ch)
		}()
		return pullStartMsg{stream: ch}
	}
}

func waitForPull(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return nil
		}
		return msg
	}
}

func (m *chatUI) handlePaste(text string) bool {
	paths, remainder, ok := parseDroppedPaths(text)
	if !ok {
		return false
	}
	added := 0
	for _, path := range paths {
		item, err := attachmentFromPath(path, m.cfg.MaxFileBytes, m.countAttachmentsByKind("image"), m.countAttachmentsByKind("file"))
		if err != nil {
			m.appendSystemMessage(err.Error())
			continue
		}
		m.attachments = append(m.attachments, item)
		added++
	}
	if added > 0 {
		if strings.TrimSpace(remainder) != "" {
			current := m.input.Value()
			if current != "" && !strings.HasSuffix(current, " ") && !strings.HasPrefix(remainder, " ") {
				current += " "
			}
			m.input.SetValue(current + strings.TrimSpace(remainder))
			m.input.CursorEnd()
		}
		m.appendSystemMessage(fmt.Sprintf("Added %d attachment(s).", added))
		return true
	}
	return false
}

func (m chatUI) countAttachmentsByKind(kind string) int {
	count := 0
	for _, item := range m.attachments {
		if item.Kind == kind {
			count++
		}
	}
	return count
}

func parseDroppedPaths(raw string) ([]string, string, bool) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return nil, "", false
	}

	rest := value
	paths := make([]string, 0)

	for {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if rest == "" {
			break
		}

		candidate, consumed, ok := consumeLeadingPath(rest)
		if !ok {
			break
		}
		paths = append(paths, candidate)
		rest = rest[consumed:]
	}

	if len(paths) == 0 {
		return nil, "", false
	}
	return paths, strings.TrimSpace(rest), true
}

func consumeLeadingPath(value string) (string, int, bool) {
	if value == "" {
		return "", 0, false
	}
	if value[0] == '"' {
		end := strings.Index(value[1:], "\"")
		if end < 0 {
			return "", 0, false
		}
		candidate := value[1 : 1+end]
		if _, err := os.Stat(candidate); err != nil {
			return "", 0, false
		}
		return candidate, end + 2, true
	}

	stop := len(value)
	for i, r := range value {
		if r == ' ' || r == '\t' || r == '\r' || r == '\n' {
			stop = i
			break
		}
	}
	candidate := strings.TrimSpace(value[:stop])
	if candidate == "" {
		return "", 0, false
	}
	if _, err := os.Stat(candidate); err != nil {
		return "", 0, false
	}
	return candidate, stop, true
}

func attachmentFromPath(path string, maxFileBytes int64, imageCount, fileCount int) (pendingAttachment, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".webp":
		return pendingAttachment{
			Kind:  "image",
			Path:  path,
			Label: fmt.Sprintf("[Image #%d]", imageCount+1),
		}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return pendingAttachment{}, err
	}
	if info.Size() > maxFileBytes {
		return pendingAttachment{}, fmt.Errorf("file too large: %s", filepath.Base(path))
	}
	if !isTextLikeFile(path) {
		return pendingAttachment{}, fmt.Errorf("unsupported file type: %s", filepath.Base(path))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return pendingAttachment{}, err
	}
	return pendingAttachment{
		Kind:        "file",
		Path:        path,
		Label:       fmt.Sprintf("[File #%d]", fileCount+1),
		FileContent: string(data),
	}, nil
}

func isTextLikeFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".csv", ".xml", ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".go", ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".sh", ".ps1", ".sql":
		return true
	default:
		return false
	}
}

func (m *chatUI) appendSystemMessage(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	m.messages = append(m.messages, uiMessage{Role: "system", Content: text})
}

func (m chatUI) View() string {
	width := max(20, m.width)
	height := max(10, m.height)
	header := "myllm"
	model := "none"
	if m.modelName != "" {
		model = m.modelName
	}
	top := fmt.Sprintf("%s  Model: %s", header, model)

	renderInput := m.input
	if prefix := strings.Join(m.attachmentLabels(), " "); prefix != "" {
		renderInput.Prompt = "> " + prefix + " "
	} else {
		renderInput.Prompt = "> "
	}
	renderInput.Width = max(8, width-lipgloss.Width(renderInput.Prompt)-1)
	inputLine := renderInput.View()

	overlayLines := m.renderOverlayLines()
	progressLine := m.renderDownloadProgress()
	reserved := 2 + len(overlayLines)
	if progressLine != "" {
		reserved++
	}
	bodyHeight := max(1, height-reserved)
	bodyLines := trimToLast(m.renderMessageLines(width), bodyHeight)

	lines := []string{top}
	lines = append(lines, bodyLines...)
	if progressLine != "" {
		lines = append(lines, progressLine)
	}
	lines = append(lines, inputLine)
	lines = append(lines, overlayLines...)
	return renderViewport(lines, width, height)
}

func (m chatUI) renderOverlayLines() []string {
	lines := []string{}
	if m.mode == modeSelectModel {
		lines = append(lines, "", emptyFallback(m.selectTitle, "Select")+":")
		for i, item := range m.selectItems {
			cursor := "  "
			if i == m.selectIndex {
				cursor = "> "
			}
			line := cursor + item
			if i < len(m.selectDescriptions) && strings.TrimSpace(m.selectDescriptions[i]) != "" {
				line += "  " + m.selectDescriptions[i]
			}
			lines = append(lines, line)
		}
	} else if m.isSlashMode() {
		lines = append(lines, "")
		lines = append(lines, m.renderSlashSuggestions()...)
	}
	return lines
}

func (m chatUI) renderMessageLines(width int) []string {
	lines := make([]string, 0, len(m.messages)*2)
	for _, msg := range m.messages {
		prefix := "Assistant"
		switch msg.Role {
		case "user":
			prefix = "You"
		case "system":
			prefix = "System"
		}
		prefix += ": "
		contentWidth := max(4, width-lipgloss.Width(prefix))
		wrapped := wrapDisplayText(msg.Content, contentWidth)
		if len(wrapped) == 0 {
			lines = append(lines, prefix)
			continue
		}
		indent := strings.Repeat(" ", lipgloss.Width(prefix))
		for i, line := range wrapped {
			if i == 0 {
				lines = append(lines, prefix+line)
			} else {
				lines = append(lines, indent+line)
			}
		}
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

func wrapDisplayText(text string, width int) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	paragraphs := strings.Split(text, "\n")
	lines := make([]string, 0, len(paragraphs))
	for _, paragraph := range paragraphs {
		if paragraph == "" {
			lines = append(lines, "")
			continue
		}
		lines = append(lines, wrapDisplayParagraph(paragraph, width)...)
	}
	return lines
}

func wrapDisplayParagraph(text string, width int) []string {
	if width <= 0 {
		return []string{text}
	}
	lines := []string{}
	var line strings.Builder
	cells := 0
	for _, r := range text {
		w := runewidth.RuneWidth(r)
		if w == 0 {
			line.WriteRune(r)
			continue
		}
		if cells > 0 && cells+w > width {
			lines = append(lines, line.String())
			line.Reset()
			cells = 0
		}
		line.WriteRune(r)
		cells += w
	}
	lines = append(lines, line.String())
	return lines
}

func renderViewport(lines []string, width, height int) string {
	if height <= 0 {
		return ""
	}
	if len(lines) > height {
		lines = lines[len(lines)-height:]
	}
	out := make([]string, 0, height)
	for _, line := range lines {
		out = append(out, fitDisplayLine(line, width))
	}
	for len(out) < height {
		out = append(out, strings.Repeat(" ", width))
	}
	return strings.Join(out, "\n")
}

func fitDisplayLine(line string, width int) string {
	if width <= 0 {
		return ""
	}
	visible := lipgloss.Width(line)
	if visible > width && !strings.Contains(line, "\x1b") {
		line = truncateDisplayText(line, width)
		visible = lipgloss.Width(line)
	}
	if visible < width {
		line += strings.Repeat(" ", width-visible)
	}
	return line
}

func truncateDisplayText(text string, width int) string {
	if width <= 0 {
		return ""
	}
	var out strings.Builder
	cells := 0
	for _, r := range text {
		w := runewidth.RuneWidth(r)
		if w == 0 {
			out.WriteRune(r)
			continue
		}
		if cells+w > width {
			break
		}
		out.WriteRune(r)
		cells += w
	}
	return out.String()
}

func (m chatUI) attachmentLabels() []string {
	labels := make([]string, 0, len(m.attachments))
	for _, item := range m.attachments {
		labels = append(labels, item.Label)
	}
	return labels
}

func (m chatUI) renderDownloadProgress() string {
	if m.downloadProgress == nil {
		return ""
	}
	return formatDownloadProgress(*m.downloadProgress)
}

func (m chatUI) isSlashMode() bool {
	return m.mode == modeChat && strings.HasPrefix(strings.TrimSpace(m.input.Value()), "/")
}

func (m chatUI) filteredCommands() []slashCommand {
	query := strings.TrimPrefix(strings.TrimSpace(m.input.Value()), "/")
	commands := append(allSlashCommands(), slashCommand{Name: "performance", Description: "配置性能、常驻和释放时间"})
	if query == "" {
		return commands
	}
	filtered := make([]slashCommand, 0, len(commands))
	for _, command := range commands {
		target := strings.ToLower(command.Name + " " + command.Description)
		if strings.Contains(target, strings.ToLower(query)) || strings.HasPrefix(strings.ToLower(command.Name), strings.ToLower(query)) {
			filtered = append(filtered, command)
		}
	}
	return filtered
}

func (m chatUI) renderSlashSuggestions() []string {
	commands := m.filteredCommands()
	if len(commands) == 0 {
		return []string{"没有匹配的命令。"}
	}
	lines := make([]string, 0, len(commands))
	for i, command := range commands {
		cursor := "  "
		if i == m.slashIndex {
			cursor = "> "
		}
		lines = append(lines, fmt.Sprintf("%s/%-8s %s", cursor, command.Name, command.Description))
	}
	return lines
}

func allSlashCommands() []slashCommand {
	commands := []slashCommand{
		{Name: "help", Description: "查看全部命令说明"},
		{Name: "model", Description: "选择当前模型"},
		{Name: "pull", Description: "下载模型到 models 目录"},
		{Name: "shell", Description: "切换 shell 工具开关"},
		{Name: "clear", Description: "清空当前对话"},
		{Name: "system", Description: "切换系统提示词文件"},
		{Name: "threads", Description: "设置 CPU 线程数"},
		{Name: "context", Description: "设置上下文长度"},
		{Name: "temp", Description: "设置温度"},
		{Name: "top-p", Description: "设置 Top-p"},
		{Name: "stats", Description: "查看当前状态"},
		{Name: "doctor", Description: "检查环境状态"},
		{Name: "exit", Description: "退出程序"},
	}
	sort.Slice(commands, func(i, j int) bool { return commands[i].Name < commands[j].Name })
	return commands
}

func trimToLast(lines []string, maxLines int) []string {
	if len(lines) <= maxLines {
		return lines
	}
	return lines[len(lines)-maxLines:]
}

func emptyFallback(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func (m chatUI) openPerformanceProfileMenu() chatUI {
	m.mode = modeSelectModel
	m.selectFlow = "perf-profile"
	m.selectTitle = "性能优化: 选择预设"
	m.selectItems = []string{"极速响应", "均衡", "省内存"}
	m.selectDescriptions = []string{
		fmt.Sprintf("线程 %d，上下文 4096，更偏向响应速度", speedProfileThreads()),
		fmt.Sprintf("线程 %d，上下文 4096，日常推荐", balancedProfileThreads()),
		fmt.Sprintf("线程 %d，上下文 2048，更省内存", memoryProfileThreads()),
	}
	m.selectIndex = 0
	m.perfDraft = performanceDraft{
		Profile:       emptyFallback(m.cfg.PerformanceProfile, "balanced"),
		Threads:       m.threads,
		ContextLength: m.contextLength,
		Resident:      m.cfg.KeepModelResident,
		IdleMinutes:   m.cfg.ResidentIdleMinutes,
		EnableMMProj:  m.cfg.EnableMMProj,
	}
	return m
}

func (m chatUI) openPerformanceResidentMenu() chatUI {
	m.mode = modeSelectModel
	m.selectFlow = "perf-resident"
	m.selectTitle = "性能优化: 模型常驻"
	m.selectItems = []string{"常驻当前模型", "按需加载"}
	m.selectDescriptions = []string{
		"第一次加载后复用，后续提问更快",
		"每次提问后释放模型，更省内存",
	}
	m.selectIndex = 0
	if !m.perfDraft.Resident {
		m.selectIndex = 1
	}
	return m
}

func (m chatUI) openPerformanceIdleMenu() chatUI {
	m.mode = modeSelectModel
	m.selectFlow = "perf-idle"
	m.selectTitle = "性能优化: 常驻释放时间"
	m.selectItems = []string{"5 分钟后释放", "15 分钟后释放", "30 分钟后释放", "60 分钟后释放", "一直常驻直到退出"}
	m.selectDescriptions = []string{
		"适合临时连续提问",
		"兼顾速度与内存",
		"更长时间保温",
		"长会话更顺手",
		"应用退出前都不释放",
	}
	m.selectIndex = 1
	switch m.perfDraft.IdleMinutes {
	case 5:
		m.selectIndex = 0
	case 15:
		m.selectIndex = 1
	case 30:
		m.selectIndex = 2
	case 60:
		m.selectIndex = 3
	case 0:
		m.selectIndex = 4
	}
	return m
}

func (m chatUI) openPerformanceMMProjMenu() chatUI {
	m.mode = modeSelectModel
	m.selectFlow = "perf-mmproj"
	m.selectTitle = "性能优化: mmproj"
	m.selectItems = []string{"开启 mmproj", "关闭 mmproj"}
	m.selectDescriptions = []string{
		"支持图片理解，但会多占一些内存",
		"关闭多模态，节省内存",
	}
	m.selectIndex = 0
	if !m.perfDraft.EnableMMProj {
		m.selectIndex = 1
	}
	return m
}

func (m chatUI) applyPerformanceDraft() chatUI {
	m.threads = m.perfDraft.Threads
	m.contextLength = m.perfDraft.ContextLength
	m.cfg.DefaultThreads = m.perfDraft.Threads
	m.cfg.DefaultContextLength = m.perfDraft.ContextLength
	m.cfg.PerformanceProfile = m.perfDraft.Profile
	m.cfg.KeepModelResident = m.perfDraft.Resident
	m.cfg.ResidentIdleMinutes = m.perfDraft.IdleMinutes
	m.cfg.EnableMMProj = m.perfDraft.EnableMMProj
	_ = saveConfig(m.layout, *m.cfg)
	sharedServer.invalidate()

	residentText := "按需加载"
	if m.perfDraft.Resident {
		if m.perfDraft.IdleMinutes > 0 {
			residentText = fmt.Sprintf("常驻，空闲 %d 分钟后释放", m.perfDraft.IdleMinutes)
		} else {
			residentText = "常驻直到退出"
		}
	}

	profileText := map[string]string{
		"speed":    "极速响应",
		"balanced": "均衡",
		"memory":   "省内存",
	}[m.perfDraft.Profile]
	if profileText == "" {
		profileText = m.perfDraft.Profile
	}
	mmprojText := "关闭"
	if m.perfDraft.EnableMMProj {
		mmprojText = "开启"
	}

	m.mode = modeChat
	m.selectTitle = ""
	m.selectItems = nil
	m.selectDescriptions = nil
	m.selectFlow = ""
	m.appendSystemMessage(fmt.Sprintf("性能优化已应用: 预设=%s, 线程=%d, 上下文=%d, 模型=%s, mmproj=%s。下次提问将复用新设置。",
		profileText,
		m.perfDraft.Threads,
		m.perfDraft.ContextLength,
		residentText,
		mmprojText,
	))
	return m
}

func speedProfileThreads() int {
	return clampInt(max(4, runtime.NumCPU()-2), 4, 12)
}

func balancedProfileThreads() int {
	return clampInt(max(4, runtime.NumCPU()/2), 4, 8)
}

func memoryProfileThreads() int {
	return clampInt(max(2, runtime.NumCPU()/3), 2, 4)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
