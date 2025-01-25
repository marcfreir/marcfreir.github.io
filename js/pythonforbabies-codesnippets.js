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
