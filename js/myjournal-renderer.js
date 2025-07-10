document.addEventListener("DOMContentLoaded", () => {
    // Load and render all journal entries
    loadJournalEntries();
});

// Configuration for journal files
const JOURNAL_CONFIG = {
    filePattern: 'my-journal{number}.txt',
    basePath: './txt/',
    maxFiles: 10, // Maximum number of files to check
    container: 'markdown-section'
};

// Main function to load all journal entries
async function loadJournalEntries() {
    const container = document.getElementById(JOURNAL_CONFIG.container);
    if (!container) {
        console.error('Container element not found');
        return;
    }

    // Clear existing content
    container.innerHTML = '';

    const journalEntries = await fetchAllJournalFiles();
    
    if (journalEntries.length === 0) {
        container.innerHTML = '<p>No journal entries found.</p>';
        return;
    }

    // Render each journal entry
    journalEntries.forEach((entry, index) => {
        const entryElement = createJournalEntryElement(entry, index);
        container.appendChild(entryElement);
    });

    // Initialize copy buttons for all code blocks
    initializeCopyButtons();
    initializeCollapsibleSections();
    initializeTaskLists();
}

// Enhanced copy button functionality
function initializeCopyButtons() {
    // Remove existing event listeners to avoid duplicates
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

// Initialize collapsible sections
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

// Initialize task lists (checkboxes)
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
    
    // Try to fetch files with different naming patterns
    const patterns = [
        'my-journal{number}.txt',
        'my-journal0{number}.txt'
    ];

    for (let i = 1; i <= JOURNAL_CONFIG.maxFiles; i++) {
        for (const pattern of patterns) {
            const filename = pattern.replace('{number}', i.toString().padStart(2, '0'));
            try {
                const content = await fetchJournalFile(filename);
                if (content) {
                    entries.push({
                        filename,
                        content,
                        index: i,
                        date: extractDateFromContent(content) || new Date().toISOString()
                    });
                    break; // Found file with this number, try next number
                }
            } catch (error) {
                // Try next pattern or number
                continue;
            }
        }
    }

    // Sort entries by date (newest first) or by index
    return entries.sort((a, b) => new Date(b.date) - new Date(a.date));
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

// Extract date from content (looks for date patterns in the first few lines)
function extractDateFromContent(content) {
    const lines = content.split('\n').slice(0, 5);
    const datePatterns = [
        /(\d{4}-\d{2}-\d{2})/,
        /(\d{2}\/\d{2}\/\d{4})/,
        /(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})/i
    ];

    for (const line of lines) {
        for (const pattern of datePatterns) {
            const match = line.match(pattern);
            if (match) {
                return new Date(match[1]).toISOString();
            }
        }
    }
    return null;
}

// Create journal entry element
// function createJournalEntryElement(entry, index) {
//     const entryDiv = document.createElement('div');
//     entryDiv.className = 'journal-entry';
//     entryDiv.id = `journal-entry-${entry.index}`;
    
//     // Extract title from content
//     const title = extractTitle(entry.content);
//     const parsedContent = parseEnhancedMarkdown(entry.content);
    
//     entryDiv.innerHTML = `
//         <div class="journal-entry-header">
//             ${title ? `<h2 class="journal-entry-title">${title}</h2>` : ''}
//             <div class="journal-entry-meta">
//                 <span class="journal-entry-date">${formatDate(entry.date)}</span>
//                 <span class="journal-entry-filename">${entry.filename}</span>
//             </div>
//         </div>
//         <div class="journal-entry-content">
//             ${parsedContent}
//         </div>
//     `;
    
//     return entryDiv;
// }

// Create journal entry element
function createJournalEntryElement(entry, index) {
    const entryDiv = document.createElement('div');
    entryDiv.className = 'journal-entry';
    entryDiv.id = `journal-entry-${entry.index}`;
    
    // Extract title from content
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

// Extract title from content (first line if it's a heading)
function extractTitle(content) {
    const firstLine = content.split('\n')[0].trim();
    const titleMatch = firstLine.match(/^#+\s*(.+)$/);
    return titleMatch ? titleMatch[1] : null;
}

// Format date for display
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

// Enhanced markdown parser with more features
function parseEnhancedMarkdown(markdown) {
    let html = markdown;

    // Escape HTML first to prevent XSS
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Pre-process: Handle code blocks first to avoid processing their content
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
    
    // Handle inline code
    const inlineCodeBlocks = [];
    html = html.replace(/`([^`]+)`/g, (match, code) => {
        const placeholder = `__INLINE_CODE_${inlineCodeBlocks.length}__`;
        inlineCodeBlocks.push(`<code class="inline-code">${code}</code>`);
        return placeholder;
    });
    
    // Headers (must be at start of line)
    html = html.replace(/^##### (.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    
    // // Code blocks (```language ... ```)
    // html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
    //     const language = lang || 'text';
    //     return `<div class="code-block-container">
    //         <button class="copy-btn" data-lang="${language}">Copy</button>
    //         <pre class="code-block"><code class="language-${language}">${escapeHtml(code.trim())}</code></pre>
    //     </div>`;
    // });
    
    // // Inline code
    // html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // // Bold and italic
    // html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    // html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    
    // // Links
    // html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // // Images
    // html = html.replace(/!\[(.+?)\]\((.+?)\)/g, '<img src="$2" alt="$1" class="journal-image">');
    
    // // Lists
    // html = html.replace(/^[\s]*[\*\-\+] (.+)$/gm, "<li>$1</li>");
    
    // // Numbered lists
    // html = html.replace(/^[\s]*\d+\. (.+)$/gm, "<li>$1</li>");
    
    // // Wrap consecutive list items in ul tags
    // html = html.replace(/(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs, "<ul>$1</ul>");
    
    // // Blockquotes
    // html = html.replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>");
    
    // // Horizontal rules
    // html = html.replace(/^---$/gm, "<hr>");
    // html = html.replace(/^\*\*\*$/gm, "<hr>");
    
    // // Tables (basic support)
    // html = html.replace(/^\|(.+)\|$/gm, (match, content) => {
    //     const cells = content.split('|').map(cell => cell.trim());
    //     const cellTags = cells.map(cell => `<td>${cell}</td>`).join('');
    //     return `<tr>${cellTags}</tr>`;
    // });
    
    // // Wrap table rows in table tags
    // html = html.replace(/(<tr>.*?<\/tr>(?:\s*<tr>.*?<\/tr>)*)/gs, "<table class='journal-table'>$1</table>");
    
    // // Math expressions
    // html = html.replace(/\$\$(.+?)\$\$/gs, "<div class='math-block'>$1</div>");
    // html = html.replace(/\$(.+?)\$/g, "<span class='math'>$1</span>");
    
    // // Strikethrough
    // html = html.replace(/~~(.+?)~~/g, "<del>$1</del>");
    
    // // Highlight
    // html = html.replace(/==(.+?)==/g, "<mark>$1</mark>");
    
    // // Line breaks and paragraphs
    // html = html.replace(/\n\n+/g, "</p><p>");
    // html = html.replace(/\n/g, "<br>");
    
    // // Wrap in paragraphs
    // html = `<p>${html}</p>`;
    
    // // Clean up empty paragraphs and fix nested tags
    // html = html.replace(/<p><\/p>/g, "");
    // html = html.replace(/<p>(<h[1-6]>.*?<\/h[1-6]>)<\/p>/g, "$1");
    // html = html.replace(/<p>(<div.*?<\/div>)<\/p>/g, "$1");
    // html = html.replace(/<p>(<table.*?<\/table>)<\/p>/g, "$1");
    // html = html.replace(/<p>(<ul.*?<\/ul>)<\/p>/g, "$1");
    // html = html.replace(/<p>(<ol.*?<\/ol>)<\/p>/g, "$1");
    // html = html.replace(/<p>(<blockquote.*?<\/blockquote>)<\/p>/g, "$1");
    // html = html.replace(/<p>(<hr>)<\/p>/g, "$1");


    // Collapsible sections (new feature)
    html = html.replace(/^<details>\s*(.+)$/gm, '<div class="collapsible-header"><span class="collapse-icon">▼</span> $1</div><div class="collapsible-content">');
    html = html.replace(/^<\/details>$/gm, '</div>');
    
    // Enhanced blockquotes with different types
    html = html.replace(/^> \[!NOTE\]\s*(.+)$/gm, '<div class="blockquote note"><strong>📝 Note:</strong> $1</div>');
    html = html.replace(/^> \[!WARNING\]\s*(.+)$/gm, '<div class="blockquote warning"><strong>⚠️ Warning:</strong> $1</div>');
    html = html.replace(/^> \[!TIP\]\s*(.+)$/gm, '<div class="blockquote tip"><strong>💡 Tip:</strong> $1</div>');
    html = html.replace(/^> \[!IMPORTANT\]\s*(.+)$/gm, '<div class="blockquote important"><strong>❗ Important:</strong> $1</div>');
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
    
    // Enhanced lists with task lists
    html = html.replace(/^[\s]*- \[ \] (.+)$/gm, '<li class="task-item"><input type="checkbox" class="task-checkbox"> $1</li>');
    html = html.replace(/^[\s]*- \[x\] (.+)$/gm, '<li class="task-item completed"><input type="checkbox" class="task-checkbox" checked> $1</li>');
    html = html.replace(/^[\s]*- \[X\] (.+)$/gm, '<li class="task-item completed"><input type="checkbox" class="task-checkbox" checked> $1</li>');
    
    // Regular unordered lists
    html = html.replace(/^[\s]*[\*\-\+] (.+)$/gm, "<li>$1</li>");
    
    // Numbered lists
    html = html.replace(/^[\s]*\d+\. (.+)$/gm, "<li>$1</li>");
    
    // Wrap consecutive list items in ul tags
    html = html.replace(/(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs, "<ul>$1</ul>");
    
    // Wrap task items in task list
    html = html.replace(/(<li class="task-item.*?<\/li>(?:\s*<li class="task-item.*?<\/li>)*)/gs, '<ul class="task-list">$1</ul>');
    
    // Enhanced text formatting
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/~~(.+?)~~/g, "<del>$1</del>");
    html = html.replace(/==(.+?)==/g, "<mark>$1</mark>");
    html = html.replace(/\+\+(.+?)\+\+/g, "<ins>$1</ins>");
    html = html.replace(/\^\^(.+?)\^\^/g, "<sup>$1</sup>");
    html = html.replace(/~~(.+?)~~/g, "<sub>$1</sub>");
    
    // Colored text (new feature)
    html = html.replace(/\{color:([^}]+)\}(.+?)\{\/color\}/g, '<span style="color: $1">$2</span>');
    
    // Badges/Tags (new feature)
    html = html.replace(/\[#([^\]]+)\]/g, '<span class="badge">$1</span>');
    
    // Links with enhanced features
    html = html.replace(/\[(.+?)\]\((.+?)\s+"(.+?)"\)/g, '<a href="$2" title="$3" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Auto-link URLs
    html = html.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Images with enhanced features
    html = html.replace(/!\[(.+?)\]\((.+?)\s+"(.+?)"\)/g, '<img src="$2" alt="$1" title="$3" class="journal-image">');
    html = html.replace(/!\[(.+?)\]\((.+?)\)/g, '<img src="$2" alt="$1" class="journal-image">');
    
    // Horizontal rules
    html = html.replace(/^---$/gm, "<hr>");
    html = html.replace(/^\*\*\*$/gm, "<hr>");
    
    // Enhanced tables
    html = html.replace(/^\|(.+)\|$/gm, (match, content) => {
        const cells = content.split('|').map(cell => cell.trim());
        const cellTags = cells.map(cell => `<td>${cell}</td>`).join('');
        return `<tr>${cellTags}</tr>`;
    });
    
    // Table headers (if second row is separator)
    html = html.replace(/(<tr>.*?<\/tr>)\s*<tr><td>[-\s:]+<\/td>(<td>[-\s:]+<\/td>)*<\/tr>/g, (match, headerRow) => {
        const newHeaderRow = headerRow.replace(/<td>/g, '<th>').replace(/<\/td>/g, '</th>');
        return newHeaderRow;
    });
    
    // Wrap table rows in table tags
    html = html.replace(/(<tr>.*?<\/tr>(?:\s*<tr>.*?<\/tr>)*)/gs, "<table class='journal-table'>$1</table>");
    
    // Math expressions (LaTeX-style)
    html = html.replace(/\$\$(.+?)\$\$/gs, "<div class='math-block'>$1</div>");
    html = html.replace(/\$(.+?)\$/g, "<span class='math'>$1</span>");
    
    // Footnotes (new feature)
    const footnotes = [];
    html = html.replace(/\[\^(\d+)\]/g, (match, num) => {
        return `<sup><a href="#footnote-${num}" class="footnote-ref">${num}</a></sup>`;
    });
    
    // Keyboard shortcuts (new feature)
    html = html.replace(/\[\[(.+?)\]\]/g, '<kbd>$1</kbd>');
    
    // Emojis shortcodes (basic set)
    const emojiMap = {
        ':smile:': '😊', ':heart:': '❤️', ':star:': '⭐', ':check:': '✅',
        ':cross:': '❌', ':fire:': '🔥', ':rocket:': '🚀', ':bulb:': '💡',
        ':warning:': '⚠️', ':info:': 'ℹ️', ':question:': '❓', ':exclamation:': '❗'
    };
    
    Object.entries(emojiMap).forEach(([shortcode, emoji]) => {
        html = html.replace(new RegExp(shortcode.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), emoji);
    });
    
    // Line breaks and paragraphs
    html = html.replace(/\n\n+/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    
    // Wrap in paragraphs
    html = `<p>${html}</p>`;
    
    // Clean up empty paragraphs and fix nested tags
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<h[1-6]>.*?<\/h[1-6]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<div.*?<\/div>)<\/p>/g, "$1");
    html = html.replace(/<p>(<table.*?<\/table>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul.*?<\/ul>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ol.*?<\/ol>)<\/p>/g, "$1");
    html = html.replace(/<p>(<blockquote.*?<\/blockquote>)<\/p>/g, "$1");
    html = html.replace(/<p>(<hr>)<\/p>/g, "$1");
    
    // Restore code blocks and inline code
    codeBlocks.forEach((block, index) => {
        html = html.replace(`__CODE_BLOCK_${index}__`, block);
    });
    
    inlineCodeBlocks.forEach((block, index) => {
        html = html.replace(`__INLINE_CODE_${index}__`, block);
    });
    
    return html;
}

// Escape HTML entities
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Utility function to refresh journal entries
function refreshJournalEntries() {
    loadJournalEntries();
}

// Export functions for external use
window.journalRenderer = {
    refresh: refreshJournalEntries,
    loadEntries: loadJournalEntries,
    config: JOURNAL_CONFIG
};