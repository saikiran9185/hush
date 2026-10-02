// ask: shows a Hush message in a small floating panel (Resolve Free scripts can't show UI).
//   ask --message "Select a clip in Resolve, then run Hush again."
import AppKit
import SwiftUI

struct MessageView: View {
    let message: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Hush", systemImage: "waveform.path").font(.headline)
            Text(message).fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("OK") { exit(0) }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 400)
    }
}

let message = CommandLine.arguments.dropFirst().last ?? ""

MainActor.assumeIsolated {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let panel = NSPanel(contentRect: .zero, styleMask: [.titled, .nonactivatingPanel, .fullSizeContentView],
                        backing: .buffered, defer: false)
    panel.titlebarAppearsTransparent = true
    panel.titleVisibility = .hidden
    panel.isMovableByWindowBackground = true
    panel.level = .floating // stays above Resolve
    panel.contentView = NSHostingView(rootView: MessageView(message: message))
    panel.setContentSize(panel.contentView!.fittingSize)
    panel.center()
    panel.orderFrontRegardless()
    DispatchQueue.main.asyncAfter(deadline: .now() + 120) { exit(0) }
    app.run()
}
