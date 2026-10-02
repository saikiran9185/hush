// soundmap: what sounds are in a recording, and when. Uses Apple's built-in on-device sound
// classifier (SoundAnalysis, 300+ sound types), so it's fast, light and needs no download.
//   soundmap file.wav   prints {"hop": 0.5, "frames": [[time, {"speech": 0.93, ...}], ...]}
import AVFoundation
import Foundation
import SoundAnalysis

final class Collector: NSObject, SNResultsObserving {
    var frames: [[Any]] = []
    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let result = result as? SNClassificationResult else { return }
        var scores: [String: Double] = [:]
        for c in result.classifications where c.confidence >= 0.05 { scores[c.identifier] = (c.confidence * 1000).rounded() / 1000 }
        frames.append([result.timeRange.start.seconds, scores])
    }
}

let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
request.windowDuration = CMTime(seconds: 1.0, preferredTimescale: 48000)
request.overlapFactor = 0.5
let analyzer = try SNAudioFileAnalyzer(url: URL(fileURLWithPath: CommandLine.arguments[1]))
let collector = Collector()
try analyzer.add(request, withObserver: collector)
analyzer.analyze()
let json = try JSONSerialization.data(withJSONObject: ["hop": 0.5, "frames": collector.frames])
FileHandle.standardOutput.write(json)
