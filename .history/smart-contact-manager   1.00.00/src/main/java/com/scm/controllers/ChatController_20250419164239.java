package com.scm.controllers;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@RestController
public class ChatController {

    @PostMapping("/chat")
    public ResponseEntity<String> chat(@RequestParam String message) {
        String chatbotApiUrl = "http://localhost:5001/chat";
        RestTemplate restTemplate = new RestTemplate();

        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("message", message);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

        ResponseEntity<String> response = restTemplate.postForEntity(chatbotApiUrl, request, String.class);
        return ResponseEntity.ok(response.getBody());
    }
}
}
