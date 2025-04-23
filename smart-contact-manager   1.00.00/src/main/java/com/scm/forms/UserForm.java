package com.scm.forms;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
@ToString
public class UserForm {
    @NotBlank(message = "Username is required")
    @Size(message = "Minimum 3 Charecters is required")
    private String name;
    @Email(message="Invalid Email Address")
    @NotBlank(message = "Email is required")
    private String email;
    @NotBlank(message = "password is required")
    private String password;
    @Size(min = 8, max = 12,message="invalid phone number")
    private String phoneNumber;
    @NotBlank(message = "about is required")
    private String about;
}
