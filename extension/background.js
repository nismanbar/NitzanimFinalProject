chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.set({
        sentinelEnabled: true,
        sentinelHistory: [],
        sentinelSettings: {
            apiUrl: 'http://localhost:5000/api/analyze',
            scanInterval: 3000,
            minRiskThreshold: 0.45
        }
    });
    console.log('Sentinel Extension installed');
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url && tab.url.includes('web.whatsapp.com')) {
        chrome.storage.local.get(['sentinelEnabled'], (result) => {
            if (result.sentinelEnabled) {
                chrome.tabs.sendMessage(tabId, { action: 'startMonitoring' });
            }
        });
    }
});