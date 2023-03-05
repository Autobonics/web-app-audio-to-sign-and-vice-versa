import SpeechRecognition from "react-speech-recognition";
export interface MicProps {
    listening: boolean;
    transcript: string;
    browserSupportsSpeechRecognition: boolean;
    resetTranscript: () => void;
}
const Mic = (props: MicProps) => {
    if (!props.browserSupportsSpeechRecognition) {
        return <span>Browser doesn't support speech recognition.</span>;
    }

    return (
        <div>
            <p>Microphone: {props.listening ? "on" : "off"}</p>
            <button onClick={() => SpeechRecognition.startListening()}>
                Start
            </button>
            <button onClick={() => SpeechRecognition.stopListening()}>
                Stop
            </button>
            <button onClick={props.resetTranscript}>Reset</button>
            <p>{props.transcript}</p>
        </div>
    );
};

export default Mic;
