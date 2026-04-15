document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('monitorToggle');
    const statusText = document.getElementById('statusText');

    chrome.storage.local.get(['sentinelEnabled', 'sentinelHistory'], (result) => {
        toggle.checked = result.sentinelEnabled !== false;
        updateStatus(result.sentinelEnabled !== false);
        updateStats(result.sentinelHistory || []);
    });

    toggle.addEventListener('change', () => {
        const enabled = toggle.checked;
        chrome.storage.local.set({ sentinelEnabled: enabled });
        updateStatus(enabled);

        // Send to all supported chat platforms
        const platforms = [
            '*://web.whatsapp.com/*',
            '*://www.instagram.com/*',
            '*://discord.com/*'
        ];
        
        platforms.forEach(p => {
            chrome.tabs.query({ url: p }, (tabs) => {
                tabs.forEach(tab => {
                    chrome.tabs.sendMessage(tab.id, {
                        action: 'toggleMonitoring',
                        enabled: enabled
                    });
                });
            });
        });
    });

    function updateStatus(enabled) {
        statusText.textContent = enabled ? 'Monitoring Active' : 'Monitoring Paused';
        statusText.className = enabled ? 'status-text active' : 'status-text';
    }

    function updateStats(history) {
        const today = new Date().setHours(0,0,0,0);
        const todayAlerts = history.filter(h => h.timestamp >= today).length;
        document.getElementById('alertCount').textContent = todayAlerts;

        if (history.length > 0) {
            const latestRisk = history[0].analysis.risk_score;
            const riskEl = document.getElementById('riskLevel');
            if (latestRisk >= 0.7) {
                riskEl.textContent = 'High';
                riskEl.parentElement.className = 'stat-box danger';
            } else if (latestRisk >= 0.45) {
                riskEl.textContent = 'Medium';
                riskEl.parentElement.className = 'stat-box warning';
            } else {
                riskEl.textContent = 'Low';
            }
        }
    }

    const viewDashboardBtn = document.getElementById('viewDashboardBtn');
    viewDashboardBtn.addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5000/dashboard' });
    });

    const testAlertBtn = document.getElementById('testAlertBtn');
    testAlertBtn.addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, { action: 'testAlert' }, (response) => {
                    if (chrome.runtime.lastError) {
                        alert('Error: Make sure you are on WhatsApp Web and the extension is loaded.');
                    } else {
                        window.close(); // Close popup to see the alert
                    }
                });
            }
        });
    });
});
