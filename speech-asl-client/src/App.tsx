import { useSpeechRecognition } from "react-speech-recognition";
import { ReactNode, useEffect, useState } from "react";
import Mic from "./components/Mic";
import "./app.scss";
interface SignGif {
    url: URL;
    text: string;
}
const App = () => {
    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition,
    } = useSpeechRecognition();
    const [signGif, setSignGif] = useState<SignGif | null>(null);
    useEffect(() => {
        (async () => {
            const url = new URL("http://localhost:5000/get_sign");
            if (transcript) {
                setSignGif(null);
                url.searchParams.append("sentence", transcript);
                console.log("Getting gif for url : ", url.href);
                setSignGif({ url: url, text: transcript });
                console.log("Setting signGif");
            }
        })();
    }, [!listening]);

    const getSign = (): ReactNode => {
        return signGif != null ? (
            <div>
                <img
                    className="sign-gif"
                    src={signGif.url.href}
                    alt="sign-gif"
                ></img>
                <h3>{signGif.text}</h3>
            </div>
        ) : (
            <h1>Loading.Please wait ..</h1>
        );
    };
    return (
        <>
            <h1>Welcome to speech to asl</h1>
            <div className="flx-container">
                <div className="flx-child">
                    <Mic
                        transcript={transcript}
                        listening={listening}
                        resetTranscript={resetTranscript}
                        browserSupportsSpeechRecognition={
                            browserSupportsSpeechRecognition
                        }
                    />
                </div>
                <div className="flx-child">
                    {!listening && transcript ? (
                        <>{getSign()}</>
                    ) : (
                        <h1>Record using microphone and wait</h1>
                    )}
                </div>
            </div>
        </>
    );
};
export default App;
