// Toggle Navbar on Small Devices
document.getElementById('menuIcon').addEventListener('click', function() {
    var navbar = document.getElementById('navbar');
    navbar.classList.toggle('active');
});


// Get references to the exit button and navigation menu
var exit = document.getElementById('exit');
var nav = document.getElementById('nav');

// Add click event listener to the exit button
exit.addEventListener('click', function(e) {
    nav.classList.add('hide-mobile'); // Hide the navigation menu
    e.preventDefault(); // Prevent default behavior (e.g., following a link)
});