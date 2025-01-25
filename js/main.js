// // Toggle Navbar on Small Devices
// document.getElementById('menuIcon').addEventListener('click', function() {
//     var navbar = document.getElementById('navbar');
//     navbar.classList.toggle('active');
// });


// // Get references to the exit button and navigation menu
// var exit = document.getElementById('exit');
// var nav = document.getElementById('nav');

// // Add click event listener to the exit button
// exit.addEventListener('click', function(e) {
//     nav.classList.add('hide-mobile'); // Hide the navigation menu
//     e.preventDefault(); // Prevent default behavior (e.g., following a link)
// });

// document.getElementById('menuIcon').addEventListener('click', function() {
//     const navbar = document.getElementById('navbar');
//     const body = document.body;

//     if (navbar.classList.contains('active')) {
//         body.style.overflow = 'hidden'; // Lock body scroll when navbar is open
//     } else {
//         body.style.overflow = 'auto'; // Restore body scroll when navbar is closed
//     }
// });

// document.addEventListener("DOMContentLoaded", () => {
//     const menuIcon = document.getElementById("menuIcon");
//     const dropdownMenu = document.getElementById("dropdownMenu");

//     menuIcon.addEventListener("click", (e) => {
//         e.preventDefault(); // Prevent default link behavior
//         dropdownMenu.classList.toggle("active"); // Toggle the active class
//     });

//     // Optional: Close the dropdown when clicking outside
//     document.addEventListener("click", (e) => {
//         if (!menuIcon.contains(e.target) && !dropdownMenu.contains(e.target)) {
//             dropdownMenu.classList.remove("active");
//         }
//     });
// });


document.addEventListener("DOMContentLoaded", () => {
    const menuIcon = document.getElementById("menuIcon");
    const dropdownMenu = document.getElementById("dropdownMenu");

    menuIcon.addEventListener("click", (e) => {
        e.preventDefault(); // Prevent default link behavior
        dropdownMenu.classList.toggle("active"); // Toggle the active class
    });

    // Optional: Close the dropdown when clicking outside
    document.addEventListener("click", (e) => {
        if (!menuIcon.contains(e.target) && !dropdownMenu.contains(e.target)) {
            dropdownMenu.classList.remove("active");
        }
    });
});





// document.addEventListener("DOMContentLoaded", () => {
//     const menuIcon = document.getElementById("menuIcon");
//     const dropdownMenu = document.getElementById("dropdownMenu");
//     const section4Link = document.getElementById("section4Link");
//     const section4Dropdown = document.getElementById("section4Dropdown");

//     // Toggle main dropdown menu
//     menuIcon.addEventListener("click", (e) => {
//         e.preventDefault();
//         dropdownMenu.classList.toggle("active");
//     });

//     // Toggle subsection dropdown for Section 4
//     section4Link.addEventListener("click", (e) => {
//         e.preventDefault();
//         e.stopPropagation(); // Prevent event from bubbling up
//         section4Dropdown.classList.toggle("active");
//     });

//     // Optional: Close menus when clicking outside
//     document.addEventListener("click", (e) => {
//         if (!menuIcon.contains(e.target) && !dropdownMenu.contains(e.target)) {
//             dropdownMenu.classList.remove("active");
//             section4Dropdown.classList.remove("active");
//         }
//     });
// });
