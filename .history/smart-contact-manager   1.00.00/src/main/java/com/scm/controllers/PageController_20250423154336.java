package com.scm.controllers;

import java.security.Principal;
import java.util.Collections;

import org.apache.logging.log4j.message.Message;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
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
import org.springframework.web.bind.annotation.ResponseBody;



@Controller
public class PageController {
    @Autowired
    private UserService userService;

    // @RequestMapping("/home")
    // public String name(Model model){
    //     System.out.println("home page handler");
    //     model.addAttribute("name","substirngs technologies");
    //     model.addAttribute("youtubeChannel","takeyouforword");
    //     model.addAttribute("githubrepo","https://www.youtube.com/watch?v=SAqi7zmW1fY&list=PPSV");

    //     return "home";
    // }

    // @RequestMapping("/about")
    // public String aboutPage() {
    //     System.out.println("about page is loading");
    //     return "about";
    // }
    
    // @RequestMapping("/services")
    // public String services() {
    //     System.out.println("services page is loading");
    //     return "services";
    // }
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

        System.out.println("rigister page is loading");
        return "register";
    }

    // @RequestMapping("/contact")
    // public String contact() {
    //     System.out.println("services page is loading");
    //     return "contact";
    // }

    @GetMapping("/")
    public String index() {
        return "redirect:/login";
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
        return "redirect:/login";
    }

    @Autowired
    private JwtUtil jwtUtil;

    @RequestMapping(value = "/generate-token", method = RequestMethod.POST)
    @ResponseBody
    public ResponseEntity<?> generateToken(@RequestParam String email, @RequestParam String password) {
        User user = userService.getUserByEmail(email);
        if (user != null && user.getPassword().equals(password)) {
            String token = jwtUtil.generateToken(email);
            return ResponseEntity.ok(Collections.singletonMap("token", token));
        }
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Invalid credentials");
    }


    // @GetMapping("/dashboard")
    // public String dashboard(Model model, Principal principal) {
    //     if(principal!=null){
    //         String userName = principal.getName(); // this is usually the email or username

    //     // If you're using a custom User entity
    //     User user = userService.getUserByEmail(userName); // adjust based on your logic

    //     model.addAttribute("loggedInUser", user);
    //     }
    //      // Now accessible in Thymeleaf

    //     return "dashboard";
    // }

    @GetMapping("/dashboard")
    public String dashboard(Model model, Principal principal) {
        if (principal != null) {
            String userName = principal.getName();
            User user = userService.getUserByEmail(userName);
            model.addAttribute("loggedInUser", user);

            // Generate JWT token
            String token = jwtUtil.generateToken(user.getEmail());
            model.addAttribute("jwtToken", token);
        }
        return "dashboard";
    }


    
}

