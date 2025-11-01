document.addEventListener("DOMContentLoaded", () => {
    loadJournalEntries();
});

// Configuration for journal files
const JOURNAL_CONFIG = {
    filePattern: 'my-journal_{date}.txt', // Changed to date pattern
    basePath: './my_journal/',
    startDate: new Date('1995-01-01'), // Start date to search from
    endDate: new Date(), // Search up to today
    container: 'markdown-section'
};

// Main function to load all journal entries
async function loadJournalEntries() {
    const container = document.getElementById(JOURNAL_CONFIG.container);
    if (!container) {
        console.error('Container element not found');
        return;
    }

    container.innerHTML = '';

    const journalEntries = await fetchAllJournalFiles();
    
    if (journalEntries.length === 0) {
        container.innerHTML = '<p>No journal entries found.</p>';
        return;
    }

    journalEntries.forEach((entry, index) => {
        const entryElement = createJournalEntryElement(entry, index);
        container.appendChild(entryElement);
    });

    initializeCopyButtons();
    initializeCollapsibleSections();
    initializeTaskLists();
}

// Enhanced copy button functionality
function initializeCopyButtons() {
    document.removeEventListener('click', handleCopyClick);
    document.addEventListener('click', handleCopyClick);
}

function handleCopyClick(e) {
    if (e.target.classList.contains('copy-btn')) {
        const codeBlock = e.target.closest('.code-block-container').querySelector('code');
        const code = codeBlock.textContent;
        
        navigator.clipboard.writeText(code).then(() => {
            const originalText = e.target.textContent;
            e.target.textContent = "Copied!";
            e.target.style.backgroundColor = "#28a745";
            
            setTimeout(() => {
                e.target.textContent = originalText;
                e.target.style.backgroundColor = "#007bff";
            }, 2000);
        }).catch((err) => {
            console.error("Failed to copy text: ", err);
            e.target.textContent = "Error";
            e.target.style.backgroundColor = "#dc3545";
            
            setTimeout(() => {
                e.target.textContent = "Copy";
                e.target.style.backgroundColor = "#007bff";
            }, 2000);
        });
    }
}

function initializeCollapsibleSections() {
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', function() {
            const content = this.nextElementSibling;
            const icon = this.querySelector('.collapse-icon');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                icon.textContent = '▼';
                this.classList.add('expanded');
            } else {
                content.style.display = 'none';
                icon.textContent = '▶';
                this.classList.remove('expanded');
            }
        });
    });
}

function initializeTaskLists() {
    document.querySelectorAll('.task-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const listItem = this.closest('li');
            if (this.checked) {
                listItem.classList.add('completed');
            } else {
                listItem.classList.remove('completed');
            }
        });
    });
}

// Generate array of dates to check
function generateDatesToCheck() {
    const dates = [];
    const current = new Date(JOURNAL_CONFIG.startDate);
    const end = new Date(JOURNAL_CONFIG.endDate);
    
    while (current <= end) {
        dates.push(new Date(current));
        current.setDate(current.getDate() + 1);
    }
    
    // Reverse to show newest first
    return dates.reverse();
}

// Format date as YYYY-MM-DD
function formatDateForFilename(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Fetch all available journal files
async function fetchAllJournalFiles() {
    const entries = [];
    const datesToCheck = generateDatesToCheck();
    
    for (const date of datesToCheck) {
        const dateStr = formatDateForFilename(date);
        const filename = `my-journal_${dateStr}.txt`;
        
        try {
            const content = await fetchJournalFile(filename);
            if (content) {
                entries.push({
                    filename,
                    content,
                    date: date.toISOString()
                });
            }
        } catch (error) {
            continue;
        }
    }

    return entries;
}

// Fetch individual journal file
async function fetchJournalFile(filename) {
    try {
        const response = await fetch(`${JOURNAL_CONFIG.basePath}${filename}`);
        if (!response.ok) {
            return null;
        }
        return await response.text();
    } catch (error) {
        return null;
    }
}

// Create journal entry element
function createJournalEntryElement(entry, index) {
    const entryDiv = document.createElement('div');
    entryDiv.className = 'journal-entry';
    entryDiv.id = `journal-entry-${formatDateForFilename(new Date(entry.date))}`;
    
    const title = extractTitle(entry.content);
    const parsedContent = parseEnhancedMarkdown(entry.content);
    
    entryDiv.innerHTML = `
        <div class="journal-entry-header">
            ${title ? `<h2 class="journal-entry-title">${title}</h2>` : ''}
            <div class="journal-entry-meta">
                <img class="profile-img journal-profile-img" src="./img/marc_freir.png" alt="Marc Freir Picture">
                <span class="journal-entry-date">${formatDate(entry.date)}</span>
            </div>
        </div>
        <div class="journal-entry-content">
            ${parsedContent}
        </div>
    `;
    
    return entryDiv;
}

function extractTitle(content) {
    const firstLine = content.split('\n')[0].trim();
    const titleMatch = firstLine.match(/^#+\s*(.+)$/);
    return titleMatch ? titleMatch[1] : null;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('sv-SE', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function parseEnhancedMarkdown(markdown) {
    let html = markdown;

    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    const codeBlocks = [];
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || 'text';
        const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
        codeBlocks.push(`<div class="code-block-container">
            <button class="copy-btn" data-lang="${language}">Copy</button>
            <pre class="code-block"><code class="language-${language}">${code.trim()}</code></pre>
        </div>`);
        return placeholder;
    });
    
    const inlineCodeBlocks = [];
    html = html.replace(/`([^`]+)`/g, (match, code) => {
        const placeholder = `__INLINE_CODE_${inlineCodeBlocks.length}__`;
        inlineCodeBlocks.push(`<code class="inline-code">${code}</code>`);
        return placeholder;
    });
    
    html = html.replace(/^##### (.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    html = html.replace(/^<details>\s*(.+)$/gm, '<div class="collapsible-header"><span class="collapse-icon">▼</span> $1</div><div class="collapsible-content">');
    html = html.replace(/^<\/details>$/gm, '</div>');
    
    html = html.replace(/^> \[!NOTE\]\s*(.+)$/gm, '<div class="blockquote note"><strong>📝 Note:</strong> $1</div>');
    html = html.replace(/^> \[!WARNING\]\s*(.+)$/gm, '<div class="blockquote warning"><strong>⚠️ Warning:</strong> $1</div>');
    html = html.replace(/^> \[!TIP\]\s*(.+)$/gm, '<div class="blockquote tip"><strong>💡 Tip:</strong> $1</div>');
    html = html.replace(/^> \[!IMPORTANT\]\s*(.+)$/gm, '<div class="blockquote important"><strong>❗ Important:</strong> $1</div>');
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
    
    html = html.replace(/^[\s]*- \[ \] (.+)$/gm, '<li class="task-item"><input type="checkbox" class="task-checkbox"> $1</li>');
    html = html.replace(/^[\s]*- \[x\] (.+)$/gm, '<li class="task-item completed"><input type="checkbox" class="task-checkbox" checked> $1</li>');
    html = html.replace(/^[\s]*- \[X\] (.+)$/gm, '<li class="task-item completed"><input type="checkbox" class="task-checkbox" checked> $1</li>');
    
    html = html.replace(/^[\s]*[\*\-\+] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/^[\s]*\d+\. (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs, "<ul>$1</ul>");
    html = html.replace(/(<li class="task-item.*?<\/li>(?:\s*<li class="task-item.*?<\/li>)*)/gs, '<ul class="task-list">$1</ul>');
    
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/~~(.+?)~~/g, "<del>$1</del>");
    html = html.replace(/==(.+?)==/g, "<mark>$1</mark>");
    html = html.replace(/\+\+(.+?)\+\+/g, "<ins>$1</ins>");
    html = html.replace(/\^\^(.+?)\^\^/g, "<sup>$1</sup>");
    
    html = html.replace(/\{color:([^}]+)\}(.+?)\{\/color\}/g, '<span style="color: $1">$2</span>');
    html = html.replace(/\[#([^\]]+)\]/g, '<span class="badge">$1</span>');
    
    html = html.replace(/\[(.+?)\]\((.+?)\s+"(.+?)"\)/g, '<a href="$2" title="$3" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    
    html = html.replace(/!\[(.+?)\]\((.+?)\s+"(.+?)"\)/g, '<img src="$2" alt="$1" title="$3" class="journal-image">');
    html = html.replace(/!\[(.+?)\]\((.+?)\)/g, '<img src="$2" alt="$1" class="journal-image">');
    
    html = html.replace(/^---$/gm, "<hr>");
    html = html.replace(/^\*\*\*$/gm, "<hr>");
    
    html = html.replace(/^\|(.+)\|$/gm, (match, content) => {
        const cells = content.split('|').map(cell => cell.trim());
        const cellTags = cells.map(cell => `<td>${cell}</td>`).join('');
        return `<tr>${cellTags}</tr>`;
    });
    
    html = html.replace(/(<tr>.*?<\/tr>)\s*<tr><td>[-\s:]+<\/td>(<td>[-\s:]+<\/td>)*<\/tr>/g, (match, headerRow) => {
        const newHeaderRow = headerRow.replace(/<td>/g, '<th>').replace(/<\/td>/g, '</th>');
        return newHeaderRow;
    });
    
    html = html.replace(/(<tr>.*?<\/tr>(?:\s*<tr>.*?<\/tr>)*)/gs, "<table class='journal-table'>$1</table>");
    html = html.replace(/\$\$(.+?)\$\$/gs, "<div class='math-block'>$1</div>");
    html = html.replace(/\$(.+?)\$/g, "<span class='math'>$1</span>");
    
    html = html.replace(/\[\^(\d+)\]/g, (match, num) => {
        return `<sup><a href="#footnote-${num}" class="footnote-ref">${num}</a></sup>`;
    });
    
    html = html.replace(/\[\[(.+?)\]\]/g, '<kbd>$1</kbd>');
    
    const emojiMap = {
        ':smile:': '😊', ':heart:': '❤️', ':star:': '⭐', ':check:': '✅',
        ':cross:': '❌', ':fire:': '🔥', ':rocket:': '🚀', ':bulb:': '💡',
        ':warning:': '⚠️', ':info:': 'ℹ️', ':question:': '❓', ':exclamation:': '❗'
    };
    
    Object.entries(emojiMap).forEach(([shortcode, emoji]) => {
        html = html.replace(new RegExp(shortcode.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), emoji);
    });
    
    html = html.replace(/\n\n+/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    html = `<p>${html}</p>`;
    
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<h[1-6]>.*?<\/h[1-6]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<div.*?<\/div>)<\/p>/g, "$1");
    html = html.replace(/<p>(<table.*?<\/table>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul.*?<\/ul>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ol.*?<\/ol>)<\/p>/g, "$1");
    html = html.replace(/<p>(<blockquote.*?<\/blockquote>)<\/p>/g, "$1");
    html = html.replace(/<p>(<hr>)<\/p>/g, "$1");
    
    codeBlocks.forEach((block, index) => {
        html = html.replace(`__CODE_BLOCK_${index}__`, block);
    });
    
    inlineCodeBlocks.forEach((block, index) => {
        html = html.replace(`__INLINE_CODE_${index}__`, block);
    });
    
    return html;
}

function refreshJournalEntries() {
    loadJournalEntries();
}

window.journalRenderer = {
    refresh: refreshJournalEntries,
    loadEntries: loadJournalEntries,
    config: JOURNAL_CONFIG
};