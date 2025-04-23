console.log("script loaded");

let currentTheme=getTheme();

document.addEventListener('DOMContentLoaded',()=>{
    changeTheme();

});

function changeTheme() {
    document.querySelector('html').classList.add(currentTheme);


    const changeThemeButton = document.querySelector('#theme_change_button');
    changeThemeButton.querySelector("span").textContent=currentTheme=="light"?"Dark":"light";

    changeThemeButton.addEventListener("click", (event) => {
        const oldTheme=currentTheme;
        console.log("change the theme");
        if(currentTheme==="dark"){
            currentTheme="light";
        }
        else{
            currentTheme="dark";
        }
        setTheme(currentTheme);
        //remove the current theme
        document.querySelector("html").classList.remove(oldTheme);
        //set the current theme
        document.querySelector("html").classList.add(currentTheme);
        changeThemeButton.querySelector("span").textContent=currentTheme=="light"?"Dark":"light";
    });
}


function setTheme(theme){
    localStorage.setItem("theme",theme);
}


function getTheme(){
    let theme=localStorage.getItem("theme");
    return theme ? theme : "light";
}