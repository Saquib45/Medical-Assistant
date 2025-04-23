package com.scm.helper;

import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.OAuth2User;

public class Helper {
    public static String getEmailOfLoggedInUser(Authentication authentication){

        //agar email ya password se login kiya hai to : email kaise nikalenge
        if(authentication instanceof OAuth2AuthenticationToken){
            
            var aOAuth2AuthenticationToken=(OAuth2AuthenticationToken)authentication;
            var clientId=aOAuth2AuthenticationToken.getAuthorizedClientRegistrationId();

            var oauth2User=(OAuth2User)authentication.getPrincipal();
            String username="";
            //sign in with google
            if(clientId.equalsIgnoreCase("google")){
                username=oauth2User.getAttribute("email").toString();
                System.out.println("getting email from google");
            }

            //sign in with github
            
            else if(clientId.equalsIgnoreCase("github")){   
                username=oauth2User.getAttribute("email")!=null?oauth2User.getAttribute("email").toString():oauth2User.getAttribute("login").toString()+"@gmail.com";
                System.out.println("getting email from github");

            }
            //sign in with facebook
            return username;
        }
        else{
            //simple singup se register kiya hai
            System.out.println("getting data from local database");
            return  authentication.getName();
        }
    }
}
