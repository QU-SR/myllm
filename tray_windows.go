//go:build windows

package main

import (
	"fmt"
	"os/exec"
	"runtime"
	"syscall"
	"unsafe"
)

const (
	createNoWindow = 0x08000000

	wmClose         = 0x0010
	wmDestroy       = 0x0002
	wmUser          = 0x0400
	wmTray          = wmUser + 1
	wmLButtonUp     = 0x0202
	wmLButtonDblClk = 0x0203
	wmRButtonUp     = 0x0205
	wmContextMenu   = 0x007B

	nimAdd     = 0x00000000
	nimDelete  = 0x00000002
	nifMessage = 0x00000001
	nifIcon    = 0x00000002
	nifTip     = 0x00000004

	mfString    = 0x00000000
	mfGrayed    = 0x00000001
	mfSeparator = 0x00000800

	tpmLeftAlign   = 0x00000000
	tpmBottomAlign = 0x00000020
	tpmRightButton = 0x00000002
	tpmReturnCmd   = 0x00000100
	tpmNonotify    = 0x00000080

	trayOpenID = 1001
	trayAPIID  = 1002
	trayExitID = 1003
)

var (
	user32  = syscall.NewLazyDLL("user32.dll")
	shell32 = syscall.NewLazyDLL("shell32.dll")
	kernel  = syscall.NewLazyDLL("kernel32.dll")

	procRegisterClassEx  = user32.NewProc("RegisterClassExW")
	procCreateWindowEx   = user32.NewProc("CreateWindowExW")
	procDefWindowProc    = user32.NewProc("DefWindowProcW")
	procDestroyWindow    = user32.NewProc("DestroyWindow")
	procPostQuitMessage  = user32.NewProc("PostQuitMessage")
	procPostMessage      = user32.NewProc("PostMessageW")
	procGetMessage       = user32.NewProc("GetMessageW")
	procTranslateMessage = user32.NewProc("TranslateMessage")
	procDispatchMessage  = user32.NewProc("DispatchMessageW")
	procLoadIcon         = user32.NewProc("LoadIconW")
	procCreatePopupMenu  = user32.NewProc("CreatePopupMenu")
	procAppendMenu       = user32.NewProc("AppendMenuW")
	procTrackPopupMenu   = user32.NewProc("TrackPopupMenu")
	procDestroyMenu      = user32.NewProc("DestroyMenu")
	procGetCursorPos     = user32.NewProc("GetCursorPos")
	procSetForeground    = user32.NewProc("SetForegroundWindow")

	procShellNotifyIcon = shell32.NewProc("Shell_NotifyIconW")
	procGetModuleHandle = kernel.NewProc("GetModuleHandleW")

	activeTrayWindow uintptr
	trayWndProc      uintptr
)

type wndClassEx struct {
	Size       uint32
	Style      uint32
	WndProc    uintptr
	ClsExtra   int32
	WndExtra   int32
	Instance   uintptr
	Icon       uintptr
	Cursor     uintptr
	Background uintptr
	MenuName   *uint16
	ClassName  *uint16
	IconSm     uintptr
}

type point struct {
	X int32
	Y int32
}

type message struct {
	HWnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      point
	Private uint32
}

type notifyIconData struct {
	Size            uint32
	HWnd            uintptr
	ID              uint32
	Flags           uint32
	CallbackMessage uint32
	Icon            uintptr
	Tip             [128]uint16
	State           uint32
	StateMask       uint32
	Info            [256]uint16
	Version         uint32
	InfoTitle       [64]uint16
	InfoFlags       uint32
	GuidItem        [16]byte
	BalloonIcon     uintptr
}

func configureDetachedProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: createNoWindow,
	}
}

func configureLauncherProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: createNoWindow,
	}
}

func runTray(layout Layout, baseURL string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	instance, _, _ := procGetModuleHandle.Call(0)
	className, _ := syscall.UTF16PtrFromString("myllmTrayWindow")

	trayWndProc = syscall.NewCallback(func(hwnd uintptr, msg uint32, wParam, lParam uintptr) uintptr {
		switch msg {
		case wmTray:
			_ = appendLog(layout, "tray-open.log", fmt.Sprintf("tray callback lParam=0x%x", lParam))
			switch uint32(lParam) {
			case wmLButtonUp, wmLButtonDblClk:
				_ = appendLog(layout, "tray-open.log", "tray icon clicked")
				go func() { _ = launchChatWindow(layout) }()
				return 0
			case wmRButtonUp, wmContextMenu:
				id := showTrayMenu(hwnd, baseURL)
				_ = appendLog(layout, "tray-open.log", fmt.Sprintf("tray menu command=%d", id))
				switch id {
				case trayOpenID:
					_ = appendLog(layout, "tray-open.log", "open chat selected")
					go func() { _ = launchChatWindow(layout) }()
				case trayExitID:
					_ = appendLog(layout, "tray-open.log", "exit background selected")
					procDestroyWindow.Call(hwnd)
				}
				return 0
			}
		case wmClose:
			procDestroyWindow.Call(hwnd)
			return 0
		case wmDestroy:
			procPostQuitMessage.Call(0)
			return 0
		}
		ret, _, _ := procDefWindowProc.Call(hwnd, uintptr(msg), wParam, lParam)
		return ret
	})

	wc := wndClassEx{
		Size:      uint32(unsafe.Sizeof(wndClassEx{})),
		WndProc:   trayWndProc,
		Instance:  instance,
		ClassName: className,
	}
	if atom, _, err := procRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc))); atom == 0 {
		return err
	}

	hwnd, _, err := procCreateWindowEx.Call(
		0,
		uintptr(unsafe.Pointer(className)),
		uintptr(unsafe.Pointer(className)),
		0,
		0, 0, 0, 0,
		0, 0,
		instance,
		0,
	)
	if hwnd == 0 {
		return err
	}
	activeTrayWindow = hwnd

	icon, _, _ := procLoadIcon.Call(0, uintptr(32512))
	nid := notifyIconData{
		Size:            uint32(unsafe.Sizeof(notifyIconData{})),
		HWnd:            hwnd,
		ID:              1,
		Flags:           nifMessage | nifIcon | nifTip,
		CallbackMessage: wmTray,
		Icon:            icon,
	}
	copy(nid.Tip[:], syscall.StringToUTF16("myllm background service\nOpenAI API: "+baseURL+"/v1"))
	if ok, _, err := procShellNotifyIcon.Call(nimAdd, uintptr(unsafe.Pointer(&nid))); ok == 0 {
		return err
	}
	defer procShellNotifyIcon.Call(nimDelete, uintptr(unsafe.Pointer(&nid)))
	defer func() { activeTrayWindow = 0 }()

	_ = appendLog(layout, "tray-open.log", "native tray ready")
	var msg message
	for {
		ret, _, err := procGetMessage.Call(uintptr(unsafe.Pointer(&msg)), 0, 0, 0)
		if int32(ret) == -1 {
			return err
		}
		if ret == 0 {
			return nil
		}
		procTranslateMessage.Call(uintptr(unsafe.Pointer(&msg)))
		procDispatchMessage.Call(uintptr(unsafe.Pointer(&msg)))
	}
}

func showTrayMenu(hwnd uintptr, baseURL string) int {
	menu, _, _ := procCreatePopupMenu.Call()
	if menu == 0 {
		return 0
	}
	defer procDestroyMenu.Call(menu)

	appendMenuString(menu, mfString, trayOpenID, "Open Chat")
	appendMenuString(menu, mfString|mfGrayed, trayAPIID, "API: "+baseURL+"/v1")
	procAppendMenu.Call(menu, mfSeparator, 0, 0)
	appendMenuString(menu, mfString, trayExitID, "Exit Background")

	var p point
	if ok, _, _ := procGetCursorPos.Call(uintptr(unsafe.Pointer(&p))); ok == 0 {
		return 0
	}
	procSetForeground.Call(hwnd)
	cmd, _, _ := procTrackPopupMenu.Call(
		menu,
		tpmBottomAlign|tpmLeftAlign|tpmRightButton|tpmReturnCmd|tpmNonotify,
		uintptr(p.X),
		uintptr(p.Y),
		0,
		hwnd,
		0,
	)
	return int(cmd)
}

func appendMenuString(menu uintptr, flags uint32, id int, title string) {
	ptr, _ := syscall.UTF16PtrFromString(title)
	procAppendMenu.Call(menu, uintptr(flags), uintptr(id), uintptr(unsafe.Pointer(ptr)))
}

func requestTrayQuit() {
	if activeTrayWindow != 0 {
		procPostMessage.Call(activeTrayWindow, wmClose, 0, 0)
	}
}
