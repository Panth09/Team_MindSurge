# Calm AI - Mental Health Support Application

## Overview
Calm AI is a web-based application designed to provide mental health support through an AI companion. The platform offers supportive, judgment-free conversations to help users navigate life's challenges anytime, anywhere.

## Features
- **AI Companion**: Chat with an intelligent AI companion designed to provide mental health support
- **Private & Secure**: User conversations are kept private and confidential
- **24/7 Availability**: Access support whenever needed, day or night
- **Personalized Support**: AI adapts to user needs and provides tailored guidance
- **Voice Interaction**: Natural voice communication with speech recognition and text-to-speech capabilities
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Mode**: Toggle between dark and light themes for comfortable viewing

## Technologies Used
- HTML5
- CSS3 (with custom variables for theming)
- JavaScript (Vanilla JS)
- Responsive design principles
- CSS animations and transitions

## Project Structure
```
calm-ai/
│
├── index.html              # Main landing page
├── ui.html                 # Chat interface page
├── blogs.html              # Blog articles page
├── suggestions.html        # Mental health suggestions page
├── about.html              # About us page
│
├── assets/
│   ├── css/
│   │   ├── style.css       # Main stylesheet
│   │   └── chat.css        # Chat interface styles
│   │
│   ├── js/
│   │   ├── main.js         # Main JavaScript functionality
│   │   ├── auth.js         # Authentication logic
│   │   ├── chat.js         # Chat functionality
│   │   └── darkmode.js     # Theme toggling
│   │
│   └── img/                # Image assets
│
└── README.md               # Project documentation
```

## Installation and Setup
1. Clone the repository:
   ```
   git clone https://github.com/your-username/calm-ai.git
   ```

2. Navigate to the project directory:
   ```
   cd calm-ai
   ```

3. Open `index.html` in your browser to view the application.

## Usage
- **Landing Page**: Visit the homepage to learn about features and testimonials
- **Chat Now**: Click the "Chat Now" button to start a conversation with the AI
- **Sign Up/Login**: Create an account or log in to save your conversation history
- **Dark/Light Mode**: Toggle between dark and light modes using the moon/sun icon

## Development
### Prerequisites
- Basic knowledge of HTML, CSS, and JavaScript
- A modern web browser
- A code editor (VS Code, Sublime Text, etc.)

### Modifying Styles
The application uses CSS variables for theming. To modify colors:
1. Open the CSS file
2. Locate the `:root` selector
3. Modify the color variables as needed

### Adding New Features
1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authentication
For demonstration purposes, the current authentication system uses hardcoded credentials:
- Email: user@example.com
- Password: password123

In a production environment, this should be replaced with a proper authentication system.

## Future Enhancements
- Integration with professional mental health resources
- Mood tracking and analytics
- Guided meditation features
- Community support groups
- Journal feature for personal reflection

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please email support@calm-ai.example.com
