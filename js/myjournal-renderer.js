document.addEventListener("DOMContentLoaded", () => {
    loadJournalEntries();
});

// Configuration for journal files
const JOURNAL_CONFIG = {
    filePattern: 'my-journal_{date}.txt',
    basePath: './my_journal/', // Matches your folder structure
    // Only check these specific dates since we know these files exist
    specificDates: [
        '2025-01-25',
        '2025-06-10',
        '2025-11-01'
    ],
    container: 'markdown-section'
};

// Main function to load all journal entries
async function loadJournalEntries() {
    const container = document.getElementById(JOURNAL_CONFIG.container);
    if (!container) {
        console.error('Container element not found');
        return;
    }

    container.innerHTML = '<p>Loading journal entries...</p>';

    const journalEntries = await fetchAllJournalFiles();
    
    if (journalEntries.length === 0) {
        container.innerHTML = '<p>No journal entries found.</p>';
        return;
    }

    // Clear loading message
    container.innerHTML = '';

    // Sort by date, newest first
    journalEntries.sort((a, b) => new Date(b.date) - new Date(a.date));

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

// Fetch all available journal files
async function fetchAllJournalFiles() {
    const entries = [];
    
    // Check specific dates we know exist
    for (const dateStr of JOURNAL_CONFIG.specificDates) {
        const filename = `my-journal_${dateStr}.txt`;
        
        try {
            const content = await fetchJournalFile(filename);
            if (content) {
                entries.push({
                    filename,
                    content,
                    date: dateStr
                });
                console.log(`Successfully loaded: ${filename}`);
            }
        } catch (error) {
            console.warn(`Could not load ${filename}:`, error);
        }
    }

    return entries;
}

// Fetch individual journal file
async function fetchJournalFile(filename) {
    const url = `${JOURNAL_CONFIG.basePath}${filename}`;
    console.log(`Attempting to fetch: ${url}`);
    
    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.warn(`File not found: ${url} (Status: ${response.status})`);
            return null;
        }
        const content = await response.text();
        console.log(`Successfully fetched ${filename}`);
        return content;
    } catch (error) {
        console.error(`Error fetching ${url}:`, error);
        return null;
    }
}

// Create journal entry element
function createJournalEntryElement(entry, index) {
    const entryDiv = document.createElement('div');
    entryDiv.className = 'journal-entry';
    entryDiv.id = `journal-entry-${entry.date}`;
    
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
    const date = new Date(dateString + 'T00:00:00');
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function parseEnhancedMarkdown(markdown) {
    let html = markdown;

    // Escape HTML
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Code blocks
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
    
    // Inline code
    const inlineCodeBlocks = [];
    html = html.replace(/`([^`]+)`/g, (match, code) => {
        const placeholder = `__INLINE_CODE_${inlineCodeBlocks.length}__`;
        inlineCodeBlocks.push(`<code class="inline-code">${code}</code>`);
        return placeholder;
    });
    
    // Headers
    html = html.replace(/^##### (.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Collapsible sections
    html = html.replace(/^<details>\s*(.+)$/gm, '<div class="collapsible-header"><span class="collapse-icon">▼</span> $1</div><div class="collapsible-content">');
    html = html.replace(/^<\/details>$/gm, '</div>');
    
    // Blockquotes
    html = html.replace(/^> \[!NOTE\]\s*(.+)$/gm, '<div class="blockquote note"><strong>📝 Note:</strong> $1</div>');
    html = html.replace(/^> \[!WARNING\]\s*(.+)$/gm, '<div class="blockquote warning"><strong>⚠️ Warning:</strong> $1</div>');
    html = html.replace(/^> \[!TIP\]\s*(.+)$/gm, '<div class="blockquote tip"><strong>💡 Tip:</strong> $1</div>');
    html = html.replace(/^> \[!IMPORTANT\]\s*(.+)$/gm, '<div class="blockquote important"><strong>❗ Important:</strong> $1</div>');
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
    
    // Task lists
    html = html.replace(/^[\s]*- \[ \] (.+)$/gm, '<li class="task-item"><input type="checkbox" class="task-checkbox"> $1</li>');
    html = html.replace(/^[\s]*- \[x\] (.+)$/gm, '<li class="task-item completed"><input type="checkbox" class="task-checkbox" checked> $1</li>');
    html = html.replace(/^[\s]*- \[X\] (.+)$/gm, '<li class="task-item completed"><input type="checkbox" class="task-checkbox" checked> $1</li>');
    
    // Lists
    html = html.replace(/^[\s]*[\*\-\+] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/^[\s]*\d+\. (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs, "<ul>$1</ul>");
    html = html.replace(/(<li class="task-item.*?<\/li>(?:\s*<li class="task-item.*?<\/li>)*)/gs, '<ul class="task-list">$1</ul>');
    
    // Text formatting
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/~~(.+?)~~/g, "<del>$1</del>");
    html = html.replace(/==(.+?)==/g, "<mark>$1</mark>");
    
    // Links
    html = html.replace(/\[(.+?)\]\((.+?)\s+"(.+?)"\)/g, '<a href="$2" title="$3" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Images
    html = html.replace(/!\[(.+?)\]\((.+?)\s+"(.+?)"\)/g, '<img src="$2" alt="$1" title="$3" class="journal-image">');
    html = html.replace(/!\[(.+?)\]\((.+?)\)/g, '<img src="$2" alt="$1" class="journal-image">');
    
    // Horizontal rules
    html = html.replace(/^---$/gm, "<hr>");
    html = html.replace(/^\*\*\*$/gm, "<hr>");
    
    // Paragraphs
    html = html.replace(/\n\n+/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    html = `<p>${html}</p>`;
    
    // Clean up
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<h[1-6]>.*?<\/h[1-6]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<div.*?<\/div>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul.*?<\/ul>)<\/p>/g, "$1");
    html = html.replace(/<p>(<hr>)<\/p>/g, "$1");
    
    // Restore code blocks
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