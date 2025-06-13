document.addEventListener("DOMContentLoaded", () => {
    // Initialize copy functionality
    initializeCopyButtons();
    
    // Load all Python code snippets from external files
    loadAllCodeSnippets();
});

/**
 * Initialize copy button functionality for all code blocks
 */
function initializeCopyButtons() {
    const copyButtons = document.querySelectorAll(".copy-btn");

    copyButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const codeBlock = button.nextElementSibling.querySelector('code');
            const code = codeBlock ? codeBlock.textContent : button.nextElementSibling.textContent;
            
            navigator.clipboard.writeText(code).then(() => {
                // Show success message
                button.textContent = "Copied!";
                button.style.backgroundColor = "#28a745";
                setTimeout(() => {
                    button.textContent = "Copy";
                    button.style.backgroundColor = "";
                }, 2000);
            }).catch((err) => {
                console.error("Failed to copy text: ", err);
                button.textContent = "Error!";
                button.style.backgroundColor = "#dc3545";
                setTimeout(() => {
                    button.textContent = "Copy";
                    button.style.backgroundColor = "";
                }, 2000);
            });
        });
    });
}

/**
 * Configuration for code snippets - maps sections to their corresponding files
 */
const codeSnippetConfig = {
    'scaled-dot-attention': './python-codes-py/scaled-dot-attention.py',
    'multi-head-attention': './python-codes-py/multi-head-attention.py',
    'positional-encoding': './python-codes-py/positional-encoding.py',
    'encoder-layer': './python-codes-py/encoder-layer.py',
    'decoder-layer': './python-codes-py/decoder-layer.py',
    'complete-transformer': './python-codes-py/complete-transformer.py',
    // Add more mappings as needed
};

/**
 * Load all code snippets from external files
 */
async function loadAllCodeSnippets() {
    // Get all code containers that need to be populated
    const codeContainers = document.querySelectorAll('.code-container');
    
    for (const container of codeContainers) {
        const codeBlock = container.querySelector('pre code');
        if (codeBlock && codeBlock.textContent.trim() === '') {
            // Find the section this code block belongs to
            const section = container.closest('section');
            const sectionId = section ? section.id : null;
            
            if (sectionId && codeSnippetConfig[sectionId]) {
                await loadCodeSnippet(codeBlock, codeSnippetConfig[sectionId]);
            }
        }
    }
    
    // Also handle the specific python-snippets section
    await fetchAndRenderPythonSnippet();
    
    // Re-initialize Prism.js syntax highlighting after loading all code
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
}

/**
 * Load a single code snippet from a file
 */
async function loadCodeSnippet(codeElement, filePath) {
    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`Failed to fetch code from ${filePath}: ${response.status}`);
        }

        const code = await response.text();
        codeElement.textContent = code;
        
        console.log(`Successfully loaded code from ${filePath}`);
        
    } catch (error) {
        console.error(`Error loading code snippet from ${filePath}:`, error);
        codeElement.textContent = `// Error loading code from ${filePath}\n// Please check if the file exists and is accessible`;
    }
}

/**
 * Your original function, enhanced with better error handling
 */
async function fetchAndRenderPythonSnippet() {
    try {
        const response = await fetch('./python-codes-txt/code-snippet-01.txt');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Failed to fetch the Python code snippet`);
        }

        const pythonCode = await response.text();
        
        // Find the python-snippets section and inject the code
        const pythonSnippetsSection = document.querySelector('#python-snippets pre code');
        if (pythonSnippetsSection) {
            pythonSnippetsSection.textContent = pythonCode;
            console.log('Successfully loaded Python snippet for #python-snippets');
        } else {
            console.warn('Could not find #python-snippets pre code element');
        }
        
    } catch (error) {
        console.error('Error in fetchAndRenderPythonSnippet:', error);
        const pythonSnippetsSection = document.querySelector('#python-snippets pre code');
        if (pythonSnippetsSection) {
            pythonSnippetsSection.textContent = `# Error loading Python code snippet\n# ${error.message}`;
        }
    }
}

/**
 * Utility function to dynamically load code by data attributes
 * Usage: Add data-code-file="path/to/file.txt" to any <code> element
 */
function loadCodeByDataAttribute() {
    const codeElements = document.querySelectorAll('code[data-code-file]');
    
    codeElements.forEach(async (codeElement) => {
        const filePath = codeElement.getAttribute('data-code-file');
        if (filePath) {
            await loadCodeSnippet(codeElement, filePath);
        }
    });
}

/**
 * Function to reload all code snippets (useful for debugging)
 */
function reloadAllCodeSnippets() {
    console.log('Reloading all code snippets...');
    loadAllCodeSnippets();
}

// Expose functions to global scope for debugging
window.reloadAllCodeSnippets = reloadAllCodeSnippets;
window.loadCodeByDataAttribute = loadCodeByDataAttribute;
