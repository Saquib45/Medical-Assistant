package com.scm.config;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;

import com.scm.services.impl.SecurityCustomUserDetailService;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Configuration
public class SecurityConfig {
    
    //user create and login using java code with in-memory service
    // @Bean
    // public UserDetailsService userDetailsService(){
    //     UserDetails user1 = User
    //         .withDefaultPasswordEncoder()
    //         .username("admin123")
    //         .password("admin123")
    //         .roles("ADMIN", "USER")
    //         .build();

    //     var inMemoryUserDetailsManager = new InMemoryUserDetailsManager(user1);  //we can pass multiple users here
    //     return inMemoryUserDetailsManager;
    // }

    @Autowired
    private SecurityCustomUserDetailService userDetailService;

    @Autowired
    private OAuthAuthenticationSuccessHandler handler;

    //configuration of authentication provider for spring security
    @Bean
    public DaoAuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider daoAuthenticationProvider = new DaoAuthenticationProvider();
        //user detail service ka object
        daoAuthenticationProvider.setUserDetailsService(userDetailService);
        //password encoder ka object
        daoAuthenticationProvider.setPasswordEncoder(passwordEncoder());
        return daoAuthenticationProvider;
    }

    // Har page ko jo security mil rahi thi wo ab specified pages ko hi milegi jo niche mention kiye hai
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity httpSecurity) throws Exception {
        httpSecurity
            .authenticationProvider(authenticationProvider())
            .authorizeHttpRequests(authorize -> {
                authorize.requestMatchers("/user/**").authenticated();  //security is applied to all pages that start from /user/.....
                authorize.anyRequest().permitAll(); //and all other pages are permitted
            });

        // agar hame kuch bhi change karna ho to ham yaha aayenge form login se related
        httpSecurity.formLogin(formLogin -> {
            formLogin.loginPage("/login");
            formLogin.loginProcessingUrl("/authenticate");

            // 👇 Redirect to /dashboard after successful login
            formLogin.defaultSuccessUrl("/dashboard", true);

            // ❌ Make sure this is commented — this was overriding the redirect
            // formLogin.successForwardUrl("/user/profile");

            formLogin.usernameParameter("email");
            formLogin.passwordParameter("password");

            // Example failure handler (optional)
            // formLogin.failureHandler(new AuthenticationFailureHandler() {
            //     @Override
            //     public void onAuthenticationFailure(HttpServletRequest request, HttpServletResponse response,
            //             AuthenticationException exception) throws IOException, ServletException {
            //         // TODO: handle failure case
            //         response.sendRedirect("/login?error=true");
            //     }
            // });

            // Example success handler (only use if needed — will override defaultSuccessUrl)
            // formLogin.successHandler(new AuthenticationSuccessHandler() {
            //     @Override
            //     public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
            //             Authentication authentication) throws IOException, ServletException {
            //         // TODO: custom logic
            //         response.sendRedirect("/dashboard");
            //     }
            // });
        });

        // CSRF disabled for development
        httpSecurity.csrf(AbstractHttpConfigurer::disable);

        // logout configuration
        httpSecurity.logout(logoutForm -> {
            logoutForm.logoutUrl("/do-logout");
            logoutForm.logoutSuccessUrl("/login?logout=true");
        });

        // OAuth2 login config
        httpSecurity.oauth2Login(oauth -> {
            oauth.loginPage("/login");
            oauth.successHandler(handler); // custom OAuth success handler
        });

        return httpSecurity.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
