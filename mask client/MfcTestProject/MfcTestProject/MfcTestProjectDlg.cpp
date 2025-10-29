#include "pch.h"                  // 또는 "stdafx.h"
#include "MfcTestProject.h"
#include "MfcTestProjectDlg.h"
#include "afxdialogex.h"

#include <winhttp.h>
#include <memory>
#include <vector>
#include <cmath>

#pragma comment(lib, "winhttp.lib")

using namespace Gdiplus;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CMfcTestProjectDlg, CDialogEx)
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    ON_WM_SIZE()
    ON_WM_CTLCOLOR()
    ON_MESSAGE(WM_STATUS_MSG, &CMfcTestProjectDlg::OnStatusMsg)
    ON_MESSAGE(WM_FRAME_MSG, &CMfcTestProjectDlg::OnFrameMsg)
    ON_WM_DESTROY()
    ON_BN_CLICKED(IDCANCEL, &CMfcTestProjectDlg::OnBnClickedCancel)
END_MESSAGE_MAP()

CMfcTestProjectDlg::CMfcTestProjectDlg(CWnd* pParent)
    : CDialogEx(IDD_MFCTESTPROJECT_DIALOG, pParent)
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

BOOL CMfcTestProjectDlg::OnInitDialog()
{
    CDialogEx::OnInitDialog();
    SetIcon(m_hIcon, TRUE);
    SetIcon(m_hIcon, FALSE);

    // GDI+ 시작
    GdiplusStartupInput gsi;
    GdiplusStartup(&m_gdiplusToken, &gsi, nullptr);

    // 폰트
    m_fontTitle.CreatePointFont(120, L"Malgun Gothic"); // 12pt
    m_fontBody.CreatePointFont(100, L"Malgun Gothic"); // 10pt
    m_fontLog.CreatePointFont(90, L"Consolas");      //  9pt

    CreateUI();
    LayoutUI();

    m_lblConn.SetWindowTextW(L"Disconnected");
    m_lblStatus.SetWindowTextW(L"—");
    AppendLog(L"[INIT] UI initialized.");

    // 자동 시작
    StartPolling();
    return TRUE;
}

void CMfcTestProjectDlg::OnDestroy()
{
    CDialogEx::OnDestroy();
    StopPolling(); // 스레드 정지 + WinHTTP 핸들 반납
    if (m_gdiplusToken) { Gdiplus::GdiplusShutdown(m_gdiplusToken); m_gdiplusToken = 0; }
}

// ──────────────────────────── UI 생성/배치 ────────────────────────────
void CMfcTestProjectDlg::CreateUI()
{
    constexpr int IDC_PIC_VIDEO = 2001;
    constexpr int IDC_BOX_RESULT = 2002;
    constexpr int IDC_LBL_STATUS = 2003;
    constexpr int IDC_BOX_CONN = 2004;
    constexpr int IDC_LBL_CONN = 2005;
    constexpr int IDC_EDIT_LOG = 2006;

    CRect rc; GetClientRect(&rc);
    const int margin = 14;
    const int rightW = 260;
    const int logH = 90;

    // 왼쪽: 영상
    CRect rVid(margin, margin, rc.Width() - rightW - 2 * margin, rc.Height() - logH - 2 * margin);
    m_picVideo.Create(L"", WS_CHILD | WS_VISIBLE | SS_BLACKRECT, rVid, this, IDC_PIC_VIDEO);

    // 오른쪽: 결과 박스
    CRect rRes(rVid.right + margin, margin, rc.right - margin, margin + 140);
    m_boxResult.Create(L"결과 표시", WS_CHILD | WS_VISIBLE | SS_CENTER, rRes, this, IDC_BOX_RESULT);
    SetBoxStyle(m_boxResult);

    // 결과 텍스트
    CRect rResText = rRes; rResText.DeflateRect(10, 28, 10, 10);
    m_lblStatus.Create(L"-", WS_CHILD | WS_VISIBLE | SS_CENTER | SS_SUNKEN, rResText, this, IDC_LBL_STATUS);
    m_lblStatus.SetFont(&m_fontTitle);

    // 오른쪽: 서버 상태 박스
    CRect rConn(rRes.left, rRes.bottom + margin, rRes.right, rRes.bottom + margin + 110);
    m_boxConn.Create(L"서버 상태", WS_CHILD | WS_VISIBLE | SS_CENTER, rConn, this, IDC_BOX_CONN);
    SetBoxStyle(m_boxConn);

    // 서버 상태 텍스트
    CRect rConnText = rConn; rConnText.DeflateRect(10, 28, 10, 10);
    m_lblConn.Create(L"-", WS_CHILD | WS_VISIBLE | SS_CENTER | SS_SUNKEN, rConnText, this, IDC_LBL_CONN);
    m_lblConn.SetFont(&m_fontBody);

    // 아래: 로그
    CRect rLog(margin, rc.bottom - logH - margin, rc.right - margin, rc.bottom - margin);
    m_editLog.Create(WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | ES_LEFT,
        rLog, this, IDC_EDIT_LOG);
    m_editLog.SetFont(&m_fontLog);
}

void CMfcTestProjectDlg::SetBoxStyle(CStatic& box)
{
    box.SetFont(&m_fontBody);
    // 필요한 경우 사용자 드로잉으로 둥근 모서리/외곽선 추가 가능
}

void CMfcTestProjectDlg::LayoutUI()
{
    if (!IsWindow(m_picVideo.m_hWnd)) return;

    CRect rc; GetClientRect(&rc);
    const int margin = 14;
    const int rightW = 260;
    const int logH = 90;

    m_picVideo.MoveWindow(margin, margin,
        rc.Width() - rightW - 2 * margin,
        rc.Height() - logH - 2 * margin);

    CRect rVid; m_picVideo.GetWindowRect(&rVid); ScreenToClient(&rVid);

    CRect rRes(rVid.right + margin, margin, rc.right - margin, margin + 140);
    m_boxResult.MoveWindow(rRes);
    CRect rResText = rRes; rResText.DeflateRect(10, 28, 10, 10);
    m_lblStatus.MoveWindow(rResText);

    CRect rConn(rRes.left, rRes.bottom + margin, rRes.right, rRes.bottom + margin + 110);
    m_boxConn.MoveWindow(rConn);
    CRect rConnText = rConn; rConnText.DeflateRect(10, 28, 10, 10);
    m_lblConn.MoveWindow(rConnText);

    m_editLog.MoveWindow(margin, rc.bottom - logH - margin,
        rc.right - margin, logH);
}

void CMfcTestProjectDlg::OnSize(UINT nType, int cx, int cy)
{
    CDialogEx::OnSize(nType, cx, cy);
    LayoutUI();
}

// ──────────────────────────── 페인팅/색상 ────────────────────────────
HBRUSH CMfcTestProjectDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
    if (pWnd->m_hWnd == m_lblStatus.m_hWnd) {
        pDC->SetBkMode(TRANSPARENT);
        pDC->SetTextColor(m_maskOn ? RGB(10, 150, 10) : RGB(200, 30, 30));
        static HBRUSH hbr = (HBRUSH)GetStockObject(HOLLOW_BRUSH);
        return hbr;
    }
    return CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);
}

void CMfcTestProjectDlg::OnPaint()
{
    if (IsIconic()) {
        CPaintDC dc(this);
        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);
        return;
    }
    CDialogEx::OnPaint();
}

HCURSOR CMfcTestProjectDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}

// ──────────────────────────── 로그 유틸 ────────────────────────────
void CMfcTestProjectDlg::AppendLog(const CString& s)
{
    if (!IsWindow(m_editLog.m_hWnd)) return;
    int len = m_editLog.GetWindowTextLengthW();
    m_editLog.SetSel(len, len);
    CString line = s + L"\r\n";
    m_editLog.ReplaceSel(line);
}

// ──────────────────────────── JPEG 렌더 ────────────────────────────
void CMfcTestProjectDlg::DrawJpegToStatic(const CByteArray& bin, CStatic& pic)
{
    if (bin.GetCount() == 0) return;

    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, bin.GetCount());
    if (!hMem) return;
    void* p = GlobalLock(hMem);
    memcpy(p, bin.GetData(), bin.GetCount());
    GlobalUnlock(hMem);

    IStream* st = nullptr;
    if (CreateStreamOnHGlobal(hMem, TRUE, &st) == S_OK) {
        Image img(st);
        CClientDC dc(&pic);
        CRect rc; pic.GetClientRect(&rc);
        Graphics g(dc);
        g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
        g.DrawImage(&img, rc.left, rc.top, rc.Width(), rc.Height());
        st->Release();
    }
    else {
        GlobalFree(hMem);
    }
}

// ──────────────────────────── JSON 파서(단순) ────────────────────────────
bool CMfcTestProjectDlg::ParseStatusJson(const CStringA& j, bool& mask, double& prob)
{
    int im = j.Find("\"mask\"");
    int ip = j.Find("\"prob\"");
    if (im < 0 || ip < 0) return false;

    int t = j.Find("true", im);
    int f = j.Find("false", im);
    if (t != -1 && (f == -1 || t < f)) mask = true;
    else if (f != -1)                  mask = false;
    else                                return false;

    int colon = j.Find(':', ip);
    if (colon < 0) return false;
    prob = atof(j.Mid(colon + 1));
    return true;
}

// ──────────────────────────── 네트워크(세션 재사용) ────────────────────────────
bool CMfcTestProjectDlg::OpenHttpOnce()
{
    if (m_hSession) return true;
    m_hSession = WinHttpOpen(L"MFCClient/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!m_hSession) return false;

    m_hConnect = WinHttpConnect(m_hSession, L"127.0.0.1", 8000, 0);
    return m_hConnect != nullptr;
}

void CMfcTestProjectDlg::CloseHttp()
{
    if (m_hConnect) { WinHttpCloseHandle(m_hConnect); m_hConnect = nullptr; }
    if (m_hSession) { WinHttpCloseHandle(m_hSession); m_hSession = nullptr; }
}

// ──────────────────────────── 폴링 시작/정지(스레드) ────────────────────────────
void CMfcTestProjectDlg::StartPolling()
{
    if (m_running) return;
    m_running = true;
    m_lblConn.SetWindowTextW(L"Connecting...");
    AppendLog(L"[NET] Start polling...");

    OpenHttpOnce(); // 세션/커넥션 1회만 열기

    m_thrStatus = std::thread(&CMfcTestProjectDlg::StatusThread, this);
    m_thrFrame = std::thread(&CMfcTestProjectDlg::FrameThread, this);
}

void CMfcTestProjectDlg::StopPolling()
{
    if (!m_running) return;
    m_running = false;

    if (m_thrStatus.joinable()) m_thrStatus.join();
    if (m_thrFrame.joinable()) m_thrFrame.join();

    CloseHttp();
    m_lblConn.SetWindowTextW(L"Disconnected");
    AppendLog(L"[NET] Stop polling.");
}

// ──────────────────────────── 상태 폴링 스레드 ────────────────────────────
void CMfcTestProjectDlg::StatusThread()
{
    while (m_running) {
        if (!m_hConnect && !OpenHttpOnce()) { ::Sleep(500); continue; }

        HINTERNET r = WinHttpOpenRequest(m_hConnect, L"GET", L"/status", nullptr,
            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        bool ok = false; CStringA body;
        if (r && WinHttpSendRequest(r, nullptr, 0, nullptr, 0, 0, 0)
            && WinHttpReceiveResponse(r, nullptr)) {
            DWORD avail = 0, read = 0;
            do {
                if (!WinHttpQueryDataAvailable(r, &avail) || !avail) break;
                std::unique_ptr<char[]> buf(new char[avail + 1]);
                if (WinHttpReadData(r, buf.get(), avail, &read) && read) {
                    buf[read] = 0; body += buf.get(); ok = true;
                }
            } while (true);
        }
        if (r) WinHttpCloseHandle(r);

        if (ok) {
            bool m = false; double p = 0.0;
            if (ParseStatusJson(body, m, p)) {
                auto* st = new StatusPayload{ m, p };
                PostMessage(WM_STATUS_MSG, 0, (LPARAM)st);
            }
            else {
                PostMessage(WM_STATUS_MSG, 0, (LPARAM)nullptr); // parse fail
            }
        }
        else {
            PostMessage(WM_STATUS_MSG, 1, (LPARAM)nullptr);     // request fail
        }

        ::Sleep(1000); // 1초 주기
    }
}

// ──────────────────────────── 프레임 폴링 스레드 ────────────────────────────
void CMfcTestProjectDlg::FrameThread()
{
    while (m_running) {
        if (!m_hConnect && !OpenHttpOnce()) { ::Sleep(200); continue; }

        HINTERNET r = WinHttpOpenRequest(m_hConnect, L"GET", L"/frame.jpg", nullptr,
            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        std::vector<BYTE>* data = nullptr;
        if (r && WinHttpSendRequest(r, nullptr, 0, nullptr, 0, 0, 0)
            && WinHttpReceiveResponse(r, nullptr)) {
            data = new std::vector<BYTE>();
            DWORD avail = 0, read = 0;
            do {
                if (!WinHttpQueryDataAvailable(r, &avail) || !avail) break;
                std::unique_ptr<BYTE[]> buf(new BYTE[avail]);
                if (WinHttpReadData(r, buf.get(), avail, &read) && read) {
                    data->insert(data->end(), buf.get(), buf.get() + read);
                }
            } while (true);
        }
        if (r) WinHttpCloseHandle(r);

        if (data && !data->empty()) {
            PostMessage(WM_FRAME_MSG, 0, (LPARAM)data);
        }
        else {
            delete data; // 실패는 조용히
        }

        ::Sleep(100); // ≈10 FPS
    }
}

// ──────────────────────────── 연결 상태 표시(디바운싱) ────────────────────────────
void CMfcTestProjectDlg::ApplyConnState()
{
    DWORD now = GetTickCount();
    bool alive = (m_lastOkTick != 0) && (now - m_lastOkTick < (DWORD)m_connHoldMs);
    m_lblConn.SetWindowTextW(alive ? L"Connected" : L"Disconnected");
}

// ──────────────────────────── UI 스레드 메시지 핸들러 ────────────────────────────
LRESULT CMfcTestProjectDlg::OnStatusMsg(WPARAM wParam, LPARAM lParam)
{
    std::unique_ptr<StatusPayload> st((StatusPayload*)lParam);

    if (st) {
        bool changed = (st->mask != m_maskOn) || (std::fabs(st->prob - m_prob) > 1e-3);
        m_maskOn = st->mask;
        m_prob = st->prob;

        m_lblStatus.SetWindowTextW(m_maskOn ? L"마스크 착용" : L"마스크 미착용");

        // ★ 최근 성공 시각 업데이트 (성공 응답 시)
        m_lastOkTick = GetTickCount();

        if (changed) InvalidateRect(nullptr, FALSE);

        CString sp; sp.Format(L"[STATUS] mask=%s prob=%.2f",
            m_maskOn ? L"ON" : L"OFF", m_prob);
        AppendLog(sp);
    }
    // wParam==1(요청 실패), st==nullptr(JSON 파싱 실패) 모두 동일하게 상태 표시 일원화
    ApplyConnState();
    return 0;
}

LRESULT CMfcTestProjectDlg::OnFrameMsg(WPARAM, LPARAM lParam)
{
    std::unique_ptr<std::vector<BYTE>> data((std::vector<BYTE>*)lParam);
    if (!data || data->empty()) {
        ApplyConnState();
        return 0;
    }

    // JPEG → 그리기
    CByteArray arr; arr.SetSize((INT_PTR)data->size());
    memcpy(arr.GetData(), data->data(), data->size());
    DrawJpegToStatic(arr, m_picVideo);

    // ★ 프레임 수신 성공 = 연결 정상 신호
    m_lastOkTick = GetTickCount();
    ApplyConnState();
    return 0;
}

void CMfcTestProjectDlg::OnBnClickedCancel()
{
    CDialogEx::OnCancel();
}
