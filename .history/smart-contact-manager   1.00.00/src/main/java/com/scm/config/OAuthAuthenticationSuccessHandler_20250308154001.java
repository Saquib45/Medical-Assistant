package com.scm.config;

import java.io.IOException;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.security.web.DefaultRedirectStrategy;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.stereotype.Component;
import java.util.List;
import com.scm.entities.Providers;
import com.scm.entities.User;
import com.scm.helper.AppConstants;
import com.scm.repositories.UserRepo;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Component
public class OAuthAuthenticationSuccessHandler implements AuthenticationSuccessHandler{

    Logger logger = LoggerFactory.getLogger(OAuthAuthenticationSuccessHandler.class);
    
    @Autowired
    private UserRepo userRepo;

    @Override
    public void onAuthenticationSuccess(
                HttpServletRequest request,
                HttpServletResponse response,
                Authentication authentication) throws IOException, ServletException {
            logger.info("OAuthAuthenticationSuccessHandler");
            DefaultOAuth2User user=(DefaultOAuth2User) authentication.getPrincipal();
            logger.info(user.getName());
            user.getAttributes().forEach((key,value)->{
                logger.info("{}=>{}",key,value);
            });
            logger.info(user.getAuthorities().toString());

            //identify provider
            var oauthe2AuthenticationToken = (OAuth2AuthenticationToken) authentication;
            String authorizedClientRegistrationId = oauthe2AuthenticationToken.getAuthorizedClientRegistrationId();
            logger.info(authorizedClientRegistrationId);
            
            var oauthUser=(DefaultOAuth2User) authentication.getPrincipal();
            oauthUser.getAttributes().forEach((key,value)->{
                //logger use karte hai server ke terminal me value print karne ke liye
                logger.info("{}=>{}",key,value);
            });

            User user1 = new User();
            //common properties
            user1.setUserId(UUID.randomUUID().toString());
            user1.setRoleList(List.of(AppConstants.ROLL_USER));
            user1.setEmailVerified(true);
            user1.setEnabled(true);
            user1.setPassword("dummy");
            //google
                //google ke property
            if(authorizedClientRegistrationId.equalsIgnoreCase("google")){
                user1.setEmail(oauthUser.getAttribute("email").toString());
                user1.setName(oauthUser.getAttribute("name").toString());
                user1.setProfilePic(oauthUser.getAttribute("picture").toString());
                user1.setProviderUserId(oauthUser.getName());
                user1.setProvider(Providers.GOOGLE);
                user1.setAbout("This account is created using google..");
            }
            //github
                //github ke property
            else if(authorizedClientRegistrationId.equalsIgnoreCase("github")){
                String email=oauthUser.getAttribute("email")!=null ? oauthUser.getAttribute("email").toString() : oauthUser.getAttribute("login").toString()+"@gmail.com";
                String picture=oauthUser.getAttribute("avatar_url").toString();
                String name=oauthUser.getAttribute("login").toString();
                String providerUserId=oauthUser.getName();

                user1.setEmail(email);
                user1.setName(name);
                user1.setProfilePic(picture);
                user1.setProviderUserId(providerUserId);
                user1.setProvider(Providers.GITHUB);
                user1.setAbout("This account is created using github..");

            }
            //facebook
                //facebook ke property
            else if(authorizedClientRegistrationId.equalsIgnoreCase("facebook")){
                
            }   
                
            //linkedin
                //linkedin ke property
            else if(authorizedClientRegistrationId.equalsIgnoreCase("linkedin")){
                
            } 

            /* 
            //data databse save
            String email=user.getAttribute("email").toString();
            String name=user.getAttribute("name").toString();
            String picture=user.getAttribute("picture").toString();

            //create user and save in database
            User user1=new User();
            user1.setEmail(email);
            user1.setName(name);
            user1.setProfilePic(picture);
            user1.setPassword("password");
            user1.setUserId(UUID.randomUUID().toString());
            user1.setProvider(Providers.GOOGLE);
            user1.setEmailVerified(true);
            user1.setProviderUserId(user.getName());
            user1.setRoleList(List.of(AppConstants.ROLL_USER));
            user1.setAbout("This account is created using google..");

            User user2=userRepo.findByEmail(email).orElse(null);

            if(user2==null){
                userRepo.save(user1);
                logger.info("User saved" + email);
            }
            */

            User user2=userRepo.findByEmail(user1.getEmail()).orElse(null);

            if(user2==null){
                userRepo.save(user1);
                logger.info("User saved" + user1.getEmail());
            }
            new DefaultRedirectStrategy().sendRedirect(request, response, "/user/profile");
       
    }

}
