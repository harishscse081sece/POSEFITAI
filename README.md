# POSEFITAI

AI-Powered Personal Fitness Trainer with Real-time Pose Detection

## 🚀 Features

- ✅ **Real-time Pose Detection** - AI tracks your exercises using MediaPipe
- ✅ **Multiple Exercises** - Push-ups, Squats, Lunges, Jumping Jacks, Shoulder Press
- ✅ **User Authentication** - Secure login/signup with JWT
- ✅ **Workout History** - Track all your workouts with detailed stats
- ✅ **Progress Dashboard** - View analytics and progress over time
- ✅ **Form Tracking** - Monitor your exercise form and posture

## 🛠️ Tech Stack

### Frontend
- React + Vite
- Tailwind CSS
- MediaPipe.js (Pose Detection)
- Axios
- React Router

### Backend
- Node.js + Express
- MongoDB + Mongoose
- JWT Authentication
- bcrypt

## 📦 Installation

### Prerequisites
- Node.js (v16+)
- MongoDB (local or Atlas)

### Backend Setup

```bash
cd backend
npm install
```

Create `.env` file:
```
MONGODB_URI=mongodb://localhost:27017/posefitai
JWT_SECRET=your_secret_key_here
PORT=5000
```

Start backend:
```bash
npm run dev
```

### Frontend Setup

```bash
cd frontend
npm install
```

Create `.env` file:
```
VITE_API_URL=http://localhost:5000/api
```

Start frontend:
```bash
npm run dev
```

## 🎯 Usage

1. Open `http://localhost:3000`
2. Sign up for an account
3. Choose an exercise from the dashboard
4. Allow webcam access
5. Start your workout!
6. View your progress in the History page

## 📱 Pages

- **Home** - Landing page with features
- **Login/Signup** - User authentication
- **Dashboard** - Exercise selection and stats
- **Workout** - Real-time exercise tracking
- **History** - Past workout records

## 🔮 Future Enhancements

- [x] Full MediaPipe integration for automatic rep counting ✅
- [ ] Leaderboard system
- [ ] Workout plans and programs
- [ ] Social features (share achievements)
- [ ] Mobile app version
- [ ] Real-time form correction feedback
- [ ] Achievement badges
- [ ] Advanced calorie tracking

## 📄 License

MIT

## 👨‍💻 Author

Built with ❤️ for fitness enthusiasts
