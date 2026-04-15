console.log('%c🛡️ SENTINEL LOADED', 'background: #1a1a2e; color: #3b82f6; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px;');
const API_URL = 'http://localhost:5000/api/analyze';
let processedMessages = new Set(); // Use a Set for long-term deduplication
let currentChatName = '';
let chatHistory = [];
let isEnabled = true;
let debugMode = true;

// Track the last seen message element to avoid re-scanning the whole DOM
let lastProcessedElement = null;

let lastElementCount = 0;

function log(msg, type = 'info') {
    if (!debugMode) return;
    const timestamp = new Date().toLocaleTimeString();
    const prefix = `[Sentinel ${timestamp}]`;
    
    switch (type) {
        case 'success': console.log(`%c${prefix} ✅ ${msg}`, 'color: #10b981; font-weight: bold;'); break;
        case 'warn': console.log(`%c${prefix} ⚠️ ${msg}`, 'color: #f59e0b; font-weight: bold;'); break;
        case 'error': console.error(`${prefix} ❌ ${msg}`); break;
        default: console.log(`%c${prefix} 🔍 ${msg}`, 'color: #3b82f6;');
    }
}

function getMessageHash(msg) {
    // Create a unique hash based on text, sender, and current chat
    return `${currentChatName}_${msg.sender}_${msg.text}`.substring(0, 255);
}

function getPlatform() {
    const host = window.location.hostname;
    if (host.includes('whatsapp')) return 'whatsapp';
    if (host.includes('instagram')) return 'instagram';
    if (host.includes('discord')) return 'discord';
    return 'unknown';
}

function getCurrentChatName(platform) {
    let name = 'Unknown Chat';
    try {
        if (platform === 'whatsapp') {
            const header = document.querySelector('header');
            if (header) {
                const titleEl = header.querySelector('span[title], div[title]');
                if (titleEl) name = titleEl.getAttribute('title') || titleEl.innerText;
            }
        } else if (platform === 'instagram') {
            const header = document.querySelector('header');
            if (header) {
                const titleEl = header.querySelector('span, div');
                if (titleEl) name = titleEl.innerText;
            }
        } else if (platform === 'discord') {
            const header = document.querySelector('section[aria-label*="Channel header"], [class*="title-"]');
            if (header) {
                const titleEl = header.querySelector('h1, h3, [class*="name-"]');
                if (titleEl) name = titleEl.innerText;
            }
        }
    } catch (e) {}
    return name.trim();
}

function extractMessages() {
    const platform = getPlatform();
    const messages = [];
    
    const newChatName = getCurrentChatName(platform);
    let chatChanged = false;
    if (newChatName !== currentChatName) {
        log(`Chat changed from "${currentChatName}" to "${newChatName}"`);
        currentChatName = newChatName;
        chatChanged = true;
    }

    let msgElements = [];
    let selectors = [];

    if (platform === 'whatsapp') {
        selectors = [
            'div[data-testid="msg-container"]',
            'div.message-in, div.message-out',
            'div[data-testid="message-bubble"]'
        ];
    } else if (platform === 'instagram') {
        selectors = [
            'div[role="none"] > div[dir="auto"]',
            'div[class*="x9f619"] div[dir="auto"]',
            'span[dir="auto"]'
        ];
    } else if (platform === 'discord') {
        selectors = [
            'div[class*="messageContent-"]',
            'li[id^="chat-messages-"]',
            'div[id^="message-content-"]'
        ];
    }

    for (const sel of selectors) {
        msgElements = document.querySelectorAll(sel);
        if (msgElements.length > 0) {
            if (msgElements.length !== lastElementCount) {
                log(`Found ${msgElements.length} messages using: ${sel} on ${platform}`, 'success');
                lastElementCount = msgElements.length;
            }
            break;
        }
    }

    if (msgElements.length === 0) {
        if (lastElementCount !== 0) {
            log(`No message elements found on ${platform}`, 'warn');
            lastElementCount = 0;
        }
        return { messages, chatChanged };
    }

    // Optimization: Only scan the last 15 elements to avoid lag
    const elementsToScan = Array.from(msgElements).slice(-15);

    elementsToScan.forEach((el) => {
        // Skip messages that are from "Yesterday" or earlier in WhatsApp
        // WhatsApp often adds "date-break" elements or spans with the date
        const container = el.closest('[data-testid="msg-container"]') || el;
        const dateHeader = container.parentElement?.querySelector('[data-testid="date-break-container"], .focusable-list-item');
        if (dateHeader && (dateHeader.innerText.includes('/') || dateHeader.innerText.toLowerCase().includes('yesterday'))) {
            return; 
        }

        let text = el.innerText.trim();
        if (!text || text.length < 2) return;

        let sender = 'Unknown';
        let isIncoming = true;

        if (platform === 'whatsapp') {
            const isOut = el.dataset.testid === 'message-out' || el.outerHTML.includes('data-testid="msg-out"');
            isIncoming = !isOut;
            sender = isIncoming ? 'Contact' : 'You';
        } else if (platform === 'instagram') {
            // Instagram usually centers messages or uses different classes for left/right
            const isOut = el.closest('div[style*="align-items: flex-end"]') !== null;
            isIncoming = !isOut;
            sender = isIncoming ? 'Instagram Contact' : 'You';
        } else if (platform === 'discord') {
            // Discord groups messages, need to find the author header
            const header = el.closest('li')?.querySelector('span[class*="username-"]');
            sender = header ? header.innerText : 'Discord User';
            isIncoming = sender !== 'You' && !header?.innerText.includes('You'); // Discord web usually doesn't show "You"
        }

        messages.push({
            text: text,
            sender: sender,
            time: new Date().toLocaleTimeString(),
            is_incoming: isIncoming,
            platform: platform
        });
    });

    return { messages, chatChanged };
}

async function analyzeMessage(msg) {
    if (!isEnabled) return null;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: msg.text,
                sender: msg.sender,
                is_incoming: msg.is_incoming,
                platform: msg.platform
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            log(`API Error (${response.status}): ${errData.error || 'Unknown error'} - ${errData.details || ''}`);
            return null;
        }

        const result = await response.json();
        return result;
    } catch (err) {
        log('Waiting for API...');
        return null;
    }
}

async function processNewMessages() {
    const { messages, chatChanged } = extractMessages();

    if (messages.length === 0) {
        log('No messages extracted');
        return;
    }

    // If chat just changed, mark ALL visible messages as processed immediately
    // to avoid treating history as "new"
    if (chatChanged) {
        log(`Chat changed to "${currentChatName}". Marking existing ${messages.length} messages as processed.`);
        messages.forEach(m => processedMessages.add(getMessageHash(m)));
        return; // Skip analysis for this cycle
    }

    // Only process messages we haven't seen yet
    const newMessages = messages.filter(m => {
        const hash = getMessageHash(m);
        if (processedMessages.has(hash)) {
            return false; // Skip - already processed
        }
        processedMessages.add(hash);
        return true;
    });

    // Limit set size to prevent memory leaks in very long sessions
    if (processedMessages.size > 1000) {
        // Clear old entries if it gets too large (simple reset for now)
        processedMessages.clear();
        log('Resetting processed messages set due to size limit.');
    }

    if (newMessages.length > 0) {
        log(`Processing ${newMessages.length} new message(s):`);
        newMessages.forEach(m => log(`  - [${m.sender}] ${m.text.substring(0, 50)}`));
    } else {
        return; // Nothing new to analyze
    }

    let chatAlreadyAlertedThisCycle = false;

    // Analyze up to 2 newest messages at a time to stay real-time
    const messagesToAnalyze = newMessages.slice(-2);

    for (const msg of messagesToAnalyze) {
        log(`Analyzing: "${msg.text.substring(0, 50)}..." from ${msg.sender}`);

        const analysis = await analyzeMessage(msg);

        if (analysis && (analysis.score !== undefined || analysis.risk_score !== undefined)) {
            const riskScore = analysis.score || analysis.risk_score;
            log(`Risk score: ${(riskScore * 100).toFixed(1)}%`);

            // Always store in history/dashboard if there is any risk
            if (riskScore > 0.45) {
                storeInHistory(msg, analysis, riskScore);
                
                // Show a warning for every NEW risky message seen
                // but only once per message (deduplication handled above)
                // and only once per 3-second cycle to prevent UI flooding
                if (!chatAlreadyAlertedThisCycle) {
                    showWarning(msg, analysis, riskScore);
                    chatAlreadyAlertedThisCycle = true;
                }
            }
        }
    }
}

function showWarning(msg, analysis, riskScore) {
    const existing = document.getElementById('sentinel-warning');
    if (existing) existing.remove();

    const riskLvl = analysis.risk_level || (riskScore >= 0.7 ? 'high' : riskScore >= 0.45 ? 'medium' : 'low');
    const warning = document.createElement('div');
    warning.id = 'sentinel-warning';
    
    // Transparent, Centered, and Smaller design
    Object.assign(warning.style, {
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%) scale(0.9)',
        zIndex: '999999',
        width: '320px',
        padding: '20px',
        borderRadius: '16px',
        // Semi-transparent background
        backgroundColor: riskLvl === 'high' ? 'rgba(254, 226, 226, 0.92)' : 'rgba(255, 251, 235, 0.92)',
        backdropFilter: 'blur(8px)',
        border: `1px solid ${riskLvl === 'high' ? 'rgba(220, 38, 38, 0.3)' : 'rgba(217, 119, 6, 0.3)'}`,
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        color: '#1f2937',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        textAlign: 'center',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        opacity: '0'
    });

    const icon = riskLvl === 'high' ? '🚨' : '⚠️';

    warning.innerHTML = `
        <div style="font-size: 32px; margin-bottom: 12px;">${icon}</div>
        <div style="font-weight: 800; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: ${riskLvl === 'high' ? '#b91c1c' : '#b45309'};">
            ${riskLvl} Risk Detected
        </div>
        
        <!-- Triggering Message Section -->
        <div style="background: rgba(255,255,255,0.5); padding: 10px; borderRadius: 8px; margin-bottom: 12px; border: 1px dashed rgba(0,0,0,0.1); font-size: 13px; color: #4b5563; font-style: italic;">
            "${msg.text.length > 80 ? msg.text.substring(0, 77) + '...' : msg.text}"
        </div>

        <div style="font-size: 14px; line-height: 1.5; font-weight: 500; margin-bottom: 16px;">
            ${analysis.recommendation}
        </div>

        <button class="sentinel-dismiss" style="width: 100%; background: ${riskLvl === 'high' ? '#ef4444' : '#f59e0b'}; color: white; border: none; padding: 10px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 13px; transition: filter 0.2s;">
            I Understand
        </button>
    `;

    warning.querySelector('.sentinel-dismiss').addEventListener('click', () => {
        warning.style.opacity = '0';
        warning.style.transform = 'translate(-50%, -50%) scale(0.8)';
        setTimeout(() => warning.remove(), 300);
    });

    document.body.appendChild(warning);
    
    // Trigger entry animation
    setTimeout(() => {
        warning.style.opacity = '1';
        warning.style.transform = 'translate(-50%, -50%) scale(1)';
    }, 10);
    
    // Auto-remove after 15 seconds
    setTimeout(() => {
        if (warning.parentElement) {
            warning.querySelector('.sentinel-dismiss').click();
        }
    }, 15000);
}

function storeInHistory(msg, analysis, riskScore) {
    chatHistory.unshift({ msg, analysis, riskScore, timestamp: Date.now() });
    if (chatHistory.length > 50) chatHistory.pop();

    chrome.storage.local.set({ sentinelHistory: chatHistory });
}

function startMonitoring() {
    log('Sentinel: Initializing monitor...');
    
    // Ignore all existing messages currently in the DOM
    const { messages, chatChanged } = extractMessages();
    messages.forEach(m => {
        const hash = getMessageHash(m);
        processedMessages.add(hash);
    });
    
    log(`Sentinel: Ignored ${messages.length} existing messages. Monitoring started.`);
    setInterval(processNewMessages, 3000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(startMonitoring, 2000);
    });
} else {
    setTimeout(startMonitoring, 2000);
}

// Force a test warning for debugging
window.sentinelTest = function() {
    log('Running manual test alert...', 'warn');
    showWarning(
        { text: 'You are so stupid and I never want to talk to you again!', sender: 'Test Contact' },
        { recommendation: 'This message contains toxic language. Consider stepping away or blocking the sender.', risk_level: 'high', pattern_label: 'toxic' },
        0.92
    );
};

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'toggleMonitoring') {
        isEnabled = request.enabled;
        sendResponse({ status: isEnabled ? 'enabled' : 'disabled' });
    }
    if (request.action === 'getHistory') {
        sendResponse({ history: chatHistory });
    }
    if (request.action === 'debug') {
        log('Debug requested');
        const { messages } = extractMessages();
        sendResponse({ messagesFound: messages.length, processedCount: processedMessages.size });
    }
    if (request.action === 'testAlert') {
        window.sentinelTest();
        sendResponse({ status: 'test_triggered' });
    }
});