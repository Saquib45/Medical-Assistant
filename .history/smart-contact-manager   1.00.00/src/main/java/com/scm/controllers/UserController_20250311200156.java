package com.scm.controllers;

import java.security.Principal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;

import com.scm.entities.User;
import com.scm.helper.Helper;
import com.scm.services.UserService;

@Controller
@RequestMapping("/user")
public class UserController {

    private Logger logger=LoggerFactory.getLogger(UserController.class);
   
    @Autowired
    private UserService userService;
   
    // @ModelAttribute
    // public void addLoggedInUserInformation(Model model,Authentication authentication){
    //     System.out.println("Adding logged in user information");
    //     String username=Helper.getEmailOfLoggedInUser(authentication);
    //     logger.info("user logged in:{}", username);
    //     //database se data ko fetch : get user from db :
    //     User user=userService.getUserByEmail(username);
    //     System.out.println(user.getName());
    //     System.out.println(user.getEmail());
    //     model.addAttribute("loggedInUser", user);
    // }

    //user dashboard page
    @RequestMapping(value = "/dashboard")
    public String userDashboard(){
        System.out.println("this is user dashboard");
        return "user/dashboard";
    }
    //profile
    @RequestMapping(value = "/profile")
    public String userProfile(Model model,Authentication authentication){
    
        

        return "user/profile";
    }
    //user add contacts page

    //user view contact

    //user edit contact

    //user delete contact

    //user search contact
}
