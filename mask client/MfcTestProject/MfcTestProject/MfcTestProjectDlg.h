#pragma once
#include <winhttp.h>    // HTTP 통신
#include <gdiplus.h>    // GDI+ (JPEG 렌더)
#include <memory>       // std::unique_ptr
#include <thread>       // 워커 스레드
#include <atomic>       // 원자 플래그
#include <vector>       // 버퍼 등 유틸
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "gdiplus.lib")

// ── 앱 전용 메시지 ──
#define WM_STATUS_MSG   (WM_APP + 1)
#define WM_FRAME_MSG    (WM_APP + 2)

// ── 상태 페이로드 ──
struct StatusPayload { bool mask; double prob; };

class CMfcTestProjectDlg : public CDialogEx
{
public:
    CMfcTestProjectDlg(CWnd* pParent = nullptr);
    enum { IDD = IDD_MFCTESTPROJECT_DIALOG };

protected:
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    afx_msg void OnSize(UINT nType, int cx, int cy);
    afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);

    // ── 워커 스레드 → UI 반영용 메시지 핸들러 ──
    afx_msg LRESULT OnStatusMsg(WPARAM, LPARAM lParam); // 상태(json) 수신
    afx_msg LRESULT OnFrameMsg(WPARAM, LPARAM lParam);  // 프레임(jpeg) 수신
    afx_msg void OnDestroy();                           // 종료 정리

    DECLARE_MESSAGE_MAP()

private:
    // ── 연결 상태 유지용 ──
    DWORD m_lastOkTick = 0;     // 마지막 성공 시각(ms)
    int   m_connHoldMs = 3000;  // 최근 성공 이후 이 ms 동안 Connected 유지
    void  ApplyConnState();     // Connected/Disconnected 표시 일원화

private:
    // ── 기본 템플릿 멤버 (아이콘) ──
    HICON m_hIcon;

    // ── UI 컨트롤 ──
    CStatic m_picVideo;    // 카메라 영상
    CStatic m_boxResult;   // 결과 박스(컨테이너 느낌)
    CStatic m_lblStatus;   // 마스크 착용/미착용
    CStatic m_boxConn;     // 서버 상태 박스
    CStatic m_lblConn;     // Connected / Disconnected
    CEdit   m_editLog;     // 하단 로그

    // ── 상태 ──
    std::atomic_bool m_running{ false };  // 스레드 루프 on/off
    bool     m_maskOn = false;
    double   m_prob = 0.0;
    CString  m_server = L"http://127.0.0.1:8000";

    // ── 네트워크 핸들 재사용 ──
    HINTERNET m_hSession = nullptr;
    HINTERNET m_hConnect = nullptr;

    // ── 워커 스레드 ──
    std::thread m_thrStatus;  // /status 폴링
    std::thread m_thrFrame;   // /frame.jpg 폴링

    // ── GDI+ ──
    ULONG_PTR m_gdiplusToken = 0;
    CFont     m_fontTitle;  // 굵은 라벨
    CFont     m_fontBody;   // 일반 라벨
    CFont     m_fontLog;    // 로그 폰트

    // ── 레이아웃 & 생성 ──
    void CreateUI();               // 컨트롤 생성(코드)
    void LayoutUI();               // 리사이즈 배치
    void SetBoxStyle(CStatic& box);// 박스 공통 스타일

    // ── 네트워크/렌더 ──
    bool HttpGetText(const wchar_t* host, INTERNET_PORT port,
        const wchar_t* path, CStringA& outBody);
    bool HttpGetBinary(const wchar_t* host, INTERNET_PORT port,
        const wchar_t* path, CByteArray& outBin);
    void DrawJpegToStatic(const CByteArray& bin, CStatic& pic);

    // ── JSON (단순) ──
    bool ParseStatusJson(const CStringA& j, bool& mask, double& prob);

    // ── 스레드 루프 ──
    void StatusThread();   // /status 폴링
    void FrameThread();    // /frame.jpg 폴링

    // ── 헬퍼 ──
    bool OpenHttpOnce();   // m_hSession/m_hConnect 준비
    void CloseHttp();      // 핸들 정리

    // ── 실행 제어(타이머 → 스레드로 대체) ──
    void StartPolling();   // 워커 시작
    void StopPolling();    // 워커 종료

    // ── 로그 ──
    void AppendLog(const CString& s); // 로그 한 줄 추가
public:
    afx_msg void OnBnClickedCancel();
};
