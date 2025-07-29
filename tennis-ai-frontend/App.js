import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// --- Constants for Drawing ---
const POSE_CONNECTIONS = [
    // Torso
    [11, 12], [11, 23], [12, 24], [23, 24],
    // Arms
    [11, 13], [13, 15], [12, 14], [14, 16],
    // Legs
    [23, 25], [25, 27], [24, 26], [26, 28]
];

// --- Helper Function to Draw the Pose ---
function drawPose(landmarks, ctx) {
    if (!landmarks || !ctx) return;

    const { width, height } = ctx.canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#61dafb'; // A nice blue color

    // Draw connection lines
    POSE_CONNECTIONS.forEach(pair => {
        const [startIdx, endIdx] = pair;
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];

        if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
            ctx.stroke();
        }
    });

    // Draw landmark points
    ctx.fillStyle = '#ff6b6b'; // A reddish color for points
    landmarks.forEach(lm => {
        if (lm.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(lm.x * width, lm.y * height, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}


// --- The Analysis Player Component ---
function AnalysisPlayer({ videoUrl, landmarks }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || !landmarks) return;

        const ctx = canvas.getContext('2d');
        let animationFrameId;

        const onFrame = () => {
            // Find the closest frame of landmarks based on video time
            const currentTime = video.currentTime;
            const totalDuration = video.duration;
            if (totalDuration > 0) {
                const progress = currentTime / totalDuration;
                const frameIndex = Math.min(
                    Math.floor(progress * landmarks.length),
                    landmarks.length - 1
                );
                
                drawPose(landmarks[frameIndex], ctx);
            }
            animationFrameId = requestAnimationFrame(onFrame);
        };

        const handleLoadedMetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        const handlePlay = () => {
            animationFrameId = requestAnimationFrame(onFrame);
        };

        const handlePauseOrEnd = () => {
            cancelAnimationFrame(animationFrameId);
        };

        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('play', handlePlay);
        video.addEventListener('pause', handlePauseOrEnd);
        video.addEventListener('ended', handlePauseOrEnd);
        video.addEventListener('seeked', onFrame); // Redraw when user seeks

        // Cleanup
        return () => {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
            video.removeEventListener('play', handlePlay);
            video.removeEventListener('pause', handlePauseOrEnd);
            video.removeEventListener('ended', handlePauseOrEnd);
            video.removeEventListener('seeked', onFrame);
            cancelAnimationFrame(animationFrameId);
        };
    }, [landmarks, videoUrl]);

    return (
        <div className="analysis-player">
            <video ref={videoRef} src={videoUrl} controls></video>
            <canvas ref={canvasRef}></canvas>
        </div>
    );
}


// --- Main App Component ---
function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewURL, setPreviewURL] = useState(null);
    const [landmarks, setLandmarks] = useState(null);
    const [feedback, setFeedback] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            const url = URL.createObjectURL(file);
            setPreviewURL(url);
            setLandmarks(null);
            setError(null);
            setFeedback([]);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a video file first.');
            return;
        }

        const formData = new FormData();
        formData.append('video', selectedFile);

        setIsLoading(true);
        setError(null);

        try {
            const response = await axios.post('http://127.0.0.1:5001/api/analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            if (response.data.landmarks && response.data.landmarks.length > 0) {
              setLandmarks(response.data.landmarks);
              setFeedback(response.data.feedback || []);
            } else {
              setError("Analysis successful, but no poses were detected in the video.");
            }
        } catch (err) {
            console.error('Error uploading file:', err);
            setError('Failed to analyze video. Make sure the backend server is running and can process the video.');
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleReset = () => {
        setLandmarks(null);
        setFeedback([]);
        setPreviewURL(null);
        setSelectedFile(null);
        setError(null);
    }

    return (
        <div className="App">
            <header className="App-header">
                <h1>🎾 AI Tennis Coach</h1>
                <p>Upload a video of your stroke to get instant feedback.</p>
            </header>
            
            <main>
                {!landmarks && (
                    <div className="upload-section">
                        <div className="controls">
                            <input type="file" accept="video/mp4,video/quicktime" onChange={handleFileChange} />
                            <button onClick={handleUpload} disabled={isLoading || !selectedFile}>
                                {isLoading ? 'Analyzing...' : 'Analyze Stroke'}
                            </button>
                        </div>
                        {error && <p className="error-message">{error}</p>}
                        {previewURL && !isLoading && (
                            <div className="video-preview">
                                <h3>Video Preview</h3>
                                <video controls src={previewURL}></video>
                            </div>
                        )}
                    </div>
                )}

                {isLoading && <div className="loader">Analyzing... This may take a moment.</div>}

                {landmarks && (
                    <div className="results-section">
                        <h2>Analysis Results</h2>
                        <AnalysisPlayer videoUrl={previewURL} landmarks={landmarks} />

                        {feedback.length > 0 && (
                            <div className="feedback-section">
                                <h3>Analysis Feedback</h3>
                                <ul>
                                    {feedback.map((item, index) => (
                                        <li key={index}>{item}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        
                        <button onClick={handleReset} className="reset-button">
                            Analyze Another Video
                        </button>
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;

