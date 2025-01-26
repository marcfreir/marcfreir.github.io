// function copyCode(button) {
//     // Get the code element inside the same <pre> block as the button
//     const codeElement = button.nextElementSibling;

//     // Get the text content of the code element
//     const codeText = codeElement.textContent;

//     // Copy the text to the clipboard
//     navigator.clipboard.writeText(codeText)
//         .then(() => {
//             // Change button text to indicate success
//             button.textContent = "Copied!";
//             setTimeout(() => {
//                 button.textContent = "Copy";
//             }, 2000); // Reset button text after 2 seconds
//         })
//         .catch((error) => {
//             console.error("Failed to copy code: ", error);
//             button.textContent = "Failed to copy";
//         });
// }

document.addEventListener("DOMContentLoaded", () => {
    const copyButtons = document.querySelectorAll(".copy-btn");

    copyButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const code = button.nextElementSibling.textContent; // Get the code from the <pre><code> block
            navigator.clipboard.writeText(code).then(() => {
                // Show a success message
                button.textContent = "Copied!";
                setTimeout(() => {
                    button.textContent = "Copy"; // Reset button text after 2 seconds
                }, 2000);
            }).catch((err) => {
                console.error("Failed to copy text: ", err);
            });
        });
    });
});


// Function to fetch and load text content
async function loadText() {
    try {
        const response = await fetch('./txt/my-journal01.txt'); // Fetch the .txt file
        if (!response.ok) {
            throw new Error('Failed to load content.txt');
        }
        const text = await response.text(); // Read the content as text
        document.getElementById('dynamic-content').textContent = text; // Insert into the paragraph
    } catch (error) {
        console.error(error);
        document.getElementById('dynamic-content').textContent = 'Failed to load content.';
    }
}

// Call the function to load the content
loadText();

// const markdownText = `
// # My Markdown Section
// This is written in **Computer Modern** font.

// ## Subheading
// Here is a [link](https://example.com).

// - Item 1
// - Item 2
// - Item 3
//         `;

//         function parseMarkdown(markdown) {
//             markdown = markdown.replace(/^### (.+)$/gm, "<h3>$1</h3>");
//             markdown = markdown.replace(/^## (.+)$/gm, "<h2>$1</h2>");
//             markdown = markdown.replace(/^# (.+)$/gm, "<h1>$1</h1>");
//             markdown = markdown.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
//             markdown = markdown.replace(/\*(.+?)\*/g, "<em>$1</em>");
//             markdown = markdown.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');
//             markdown = markdown.replace(/^- (.+)$/gm, "<li>$1</li>");
//             markdown = markdown.replace(/<li>(.+)<\/li>/gms, "<ul>$&</ul>");
//             return markdown;
//         }

//         document.getElementById("markdown-section").innerHTML = parseMarkdown(markdownText);
   

async function fetchAndRenderMarkdown() {
    try {
        // Fetch the markdown content from a TXT file
        const response = await fetch('./txt/my-journal01.txt');
        if (!response.ok) {
            throw new Error('Failed to fetch the markdown file');
        }

        const markdownText = await response.text();

        // Parse the markdown content
        const parsedMarkdown = parseMarkdown(markdownText);

        // Render it into the markdown section
        document.getElementById('markdown-section').innerHTML = parsedMarkdown;
    } catch (error) {
        console.error(error);
        document.getElementById('markdown-section').innerHTML = 'Error loading markdown content.';
    }
}

function parseMarkdown(markdown) {
    markdown = markdown.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    markdown = markdown.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    markdown = markdown.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    markdown = markdown.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    markdown = markdown.replace(/\*(.+?)\*/g, "<em>$1</em>");
    markdown = markdown.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');
    markdown = markdown.replace(/^- (.+)$/gm, "<li>$1</li>");
    markdown = markdown.replace(/<li>(.+)<\/li>/gms, "<ul>$&</ul>");
    // Handle inline math: $...$
    markdown = markdown.replace(/\$(.+?)\$/g, "<span class='math'>$1</span>");

    // Handle block math: $$...$$
    markdown = markdown.replace(/\$\$(.+?)\$\$/gs, "<div class='math-block'>$1</div>");

    return markdown;
}

// Fetch and render the markdown content on page load
fetchAndRenderMarkdown();