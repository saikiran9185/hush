// ask: the one window Hush needs.
// Resolve Free scripts can't show UI, and a plain osascript dialog can't take the keyboard while
// Resolve is in front, so this is a tiny non-activating panel that can.
//   ask "MVI_9065 · 2.1 s"         prints "remove<TAB>car horn" or "keep<TAB>voice"; exit 1 = cancelled
//   ask --message "Some text"      just an OK button
import AppKit
import SwiftUI

final class KeyPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

struct AskView: View {
    let subtitle: String
    let message: String?
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Hush", systemImage: "waveform.path").font(.headline)
                Spacer()
                Text(subtitle).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            if let message {
                Text(message).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Spacer()
                    Button("OK") { exit(0) }.keyboardShortcut(.defaultAction)
                }
            } else {
                Text("What is this sound?").font(.title3.bold())
                TextField("car horn, dog barking, wind, traffic…", text: $text)
                    .textFieldStyle(.roundedBorder)
                    .font(.title3)
                    .focused($focused)
                    .onSubmit { answer("remove") }
                HStack {
                    Button("Cancel") { exit(1) }.keyboardShortcut(.cancelAction)
                    Spacer()
                    Button("Keep only this") { answer("keep") }.disabled(name.isEmpty)
                    Button("Remove") { answer("remove") }.keyboardShortcut(.defaultAction).disabled(name.isEmpty)
                }
            }
        }
        .padding(20)
        .frame(width: 420)
        .onAppear { focused = true }
    }

    private var name: String { text.trimmingCharacters(in: .whitespaces) }

    private func answer(_ mode: String) {
        guard !name.isEmpty else { return }
        print("\(mode)\t\(name)")
        exit(0)
    }
}

let args = Array(CommandLine.arguments.dropFirst())
let message = args.first == "--message" ? args.dropFirst().first : nil

MainActor.assumeIsolated {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let panel = KeyPanel(contentRect: .zero, styleMask: [.titled, .nonactivatingPanel, .fullSizeContentView],
                         backing: .buffered, defer: false)
    panel.titlebarAppearsTransparent = true
    panel.titleVisibility = .hidden
    panel.isMovableByWindowBackground = true
    panel.level = .floating
    panel.contentView = NSHostingView(rootView: AskView(subtitle: message == nil ? args.first ?? "" : "", message: message))
    panel.setContentSize(panel.contentView!.fittingSize)
    panel.center()
    // Non-activating: the panel gets the keyboard even though Resolve stays the active app.
    panel.makeKeyAndOrderFront(nil)
    DispatchQueue.main.asyncAfter(deadline: .now() + 300) { exit(1) } // nobody answered
    app.run()
}
