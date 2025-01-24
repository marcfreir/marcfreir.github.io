/* 
    Created on : 15-Jul-2018, 06:02:39 PM
    Author     : Marc Freir
    License    : This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
*/

//// Alert test 
//alert("Hey, baby!");

/* Variables */
// canvas variable --> this means that the document will get the object canvas with the Id="canvas"
var cvs = document.getElementById("canvas");
// context variable --> rhis means the the canvas will get the context in 2d
var ctx = cvs.getContext("2d");

// images variables
// cat.png (the cat - itself)
var cat = new Image();
// bg.png (the background)
var bg = new Image();
// fg.png (the foreground)
var fg = new Image();
// wallNorth.png (the wall from the top)
var wallNorth = new Image();
// wallSouth.png (teh wall from the bottom)
var wallSouth = new Image();

/* The source of the variables (images)*/
cat.src = "images/cat.png";
//cat.src = "images/cat.svg";
bg.src = "images/bg.png";
fg.src = "images/fg.png";
wallNorth.src = "images/wallNorth.png";
//wallNorth.src = "images/wallNorth.svg";
wallSouth.src = "images/wallSouth.png";
//wallSouth.src = "images/wallSouth.svg";

/* Other variables */
//
var gap = 150;
//
//var constant = wallNorth.height+gap;
var constant;
// cat X (X position)
var cX = 100;
// cat Y (Y position)
var cY = 150;
//
var gravity = 1.0;
var score = 0;

/* Sounds - Audio Files */
// cat flying
var fly = new Audio();
// score increment
var scor = new Audio();
// background song
var bgsong = new Audio();

/* The source of the variables (sounds)*/
fly.src = "sounds/fly.mp3";
scor.src = "sounds/score.mp3";
bgsong.src = "sounds/cute.mp3";

// backgroud song
bgsong.addEventListener("ended", function()
{
    this.currentTime = 0;
    this.play();
}, false);
bgsong.play();
//
//var stopbtn = document.getElementById("buttonstop");
// this function will pause the background song
function pauseSong()
{
    bgsong.pause();
}
// this function will play (again) the background song - because the song plays by default
function playSong()
{
    bgsong.play();
}

/* Controller */
// on key down
document.addEventListener("keydown", moveUp);
// creating the function (method) moveUp
function moveUp()
{
    cY -= 20;
    fly.play();
}

//
document.addEventListener("keypress", moveDown)
function moveDown()
{
    cY += 30;
    fly.play();
}

// wall coordinates
var wall = [];
wall[0] = { x : cvs.width, y : 0 };

/* Draw images */
// creating the function (method) draw
function draw()
{
    ctx.drawImage(bg, 0, 0);
    // for loop to draw the walls
    for(var i = 0; i < wall.length; i++)
    {
        constant = wallNorth.height+gap;
        ctx.drawImage(wallNorth, wall[i].x, wall[i].y);
        ctx.drawImage(wallSouth, wall[i].x, wall[i].y+constant);
        wall[i].x--;

        if(wall[i].x == 125)
        {
            wall.push({ x : cvs.width, y : Math.floor(Math.random()*wallNorth.height) - wallNorth.height}); 
        }

        // detect collision
        if(cX + cat.width >= wall[i].x && cX <= wall[i].x + wallNorth.width && (cY <= wall[i].y + wallNorth.height || cY + cat.height >= wall[i].y+constant) || cY + cat.height >= cvs.height - fg.height)
        {
            // reload the page
            //bump.play();
            //location.reload();
            
            bgsong.pause();
            alert("You, bad cat!");
            //stop();
            location.reload();
        }
        //
        if(wall[i].x == 5)
        {
            score++;
            scor.play();

            /*if(score == 2)
            {
                //stop();
                alert("You, beautiful cat!");
            }*/
            //
            if(score % 5 == 0)
            {
                alert("You, beautiful cat! - 2 EXTRA POINTS FOR YOU, Baby!");
                score = score + 2;
            }
        }
    }
    ctx.drawImage(fg, 0, cvs.height - fg.height);
    ctx.drawImage(cat, cX, cY);

    cY += gravity;

    ctx.fillStyle = "#000";
    ctx.font = "20px Verdana";
    ctx.fillText("Score: " +score, 10, cvs.height-20);

    requestAnimationFrame(draw);
}
// calling the function (method)
draw();