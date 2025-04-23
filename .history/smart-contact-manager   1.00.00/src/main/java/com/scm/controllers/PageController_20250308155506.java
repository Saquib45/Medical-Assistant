package com.scm.controllers;

import org.apache.logging.log4j.message.Message;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.service.annotation.GetExchange;

import com.scm.entities.User;
import com.scm.forms.UserForm;
import com.scm.helper.message;
import com.scm.helper.messageType;
import com.scm.services.UserService;

import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;



@Controller
public class PageController {
    @Autowired
    private UserService userService;

    @RequestMapping("/home")
    public String name(Model model){
        System.out.println("home page handler");
        model.addAttribute("name","substirngs technologies");
        model.addAttribute("youtubeChannel","takeyouforword");
        model.addAttribute("githubrepo","https://www.youtube.com/watch?v=SAqi7zmW1fY&list=PPSV");

        return "home";
    }

    @RequestMapping("/about")
    public String aboutPage() {
        System.out.println("about page is loading");
        return "about";
    }
    
    @RequestMapping("/services")
    public String services() {
        System.out.println("services page is loading");
        return "services";
    }
    //this is showing login page
    @RequestMapping("/login")
    public String login() {
        System.out.println("services page is loading");
        return "login";
    }
    //registration page ke liye
    @RequestMapping("/register")
    public String register(Model model) {
        UserForm userForm= new UserForm();
        //default data bhi daal sakte hai
        //userForm.setName("sakib");
        //userForm.setAbout("hi this is fun");
        model.addAttribute("userForm", userForm);

        System.out.println("services page is loading");
        return "register";
    }

    @RequestMapping("/contact")
    public String contact() {
        System.out.println("services page is loading");
        return "contact";
    }

    @GetMapping("/")
    public String index() {
        return "redirect:/home";
    }


    //controller for singup
    //registration processe ke liya
    @RequestMapping(value = "/do-register", method = RequestMethod.POST)
    public String processRegister(@Valid @ModelAttribute UserForm userForm, BindingResult rBindingResult, HttpSession session) {
        System.out.println("Processing registration");
        //fetch form data
        //for fetching make UserForm
        System.out.println(userForm);
        //validate form data(TODO in next video)
        if(rBindingResult.hasErrors()){
            return "register";
        }
        //save to database

        // we conveert UserFOrm---->user at below stage
        User user=new User();
        user.setName(userForm.getName());
        user.setEmail(userForm.getEmail());
        user.setPassword(userForm.getPassword());
        user.setAbout(userForm.getAbout());
        user.setPhoneNumber(userForm.getPhoneNumber());
        user.setProfilePic("https://images.app.goo.gl/x4RgffiqvH3gZXrw6");

        User savedUser = userService.saveUser(user);
        System.out.println("user saved");
        //message="registration successful"
        

        //add message
       message message1 = message.builder().content("Registration Successful").type(messageType.green).build();
        session.setAttribute("message", message1);

        //redirectto login page
        return "redirect:/register";
    }
    
}
