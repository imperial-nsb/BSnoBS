use std::sync::mpsc::{Receiver, channel};
use std::thread;

use axoupdater::{AxoUpdater, ReleaseSource, ReleaseSourceType, Version};
use axoupdater::errors::AxoupdateError;

const GITHUB_OWNER: &str = "imperial-nsb";
const GITHUB_REPO: &str = "BSnoBS";
const APP_NAME: &str = "bsnobs";

#[derive(Clone, Debug)]
pub enum UpdateMessage {
    UpToDate,
    Available { latest: String },
    Installed { new_version: String },
    NoUpdateApplied,
    Error(String),
}

#[derive(Default)]
pub struct UpdateUi {
    pub busy: bool,
    pub last: Option<UpdateMessage>,
    pub available: Option<String>,
    rx: Option<Receiver<UpdateMessage>>,
}

impl UpdateUi {
    pub fn current_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    pub fn poll(&mut self) -> bool {
        let Some(rx) = self.rx.as_ref() else {
            return false;
        };
        match rx.try_recv() {
            Ok(msg) => {
                self.busy = false;
                match &msg {
                    UpdateMessage::Available { latest } => {
                        self.available = Some(latest.clone());
                    }
                    UpdateMessage::Installed { .. } | UpdateMessage::NoUpdateApplied => {
                        self.available = None;
                    }
                    _ => {}
                }
                self.last = Some(msg);
                self.rx = None;
                true
            }
            Err(_) => false,
        }
    }

    pub fn check(&mut self) {
        if self.busy {
            return;
        }
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.busy = true;
        self.last = None;
        thread::spawn(move || {
            let _ = tx.send(run_check());
        });
    }

    pub fn install(&mut self) {
        if self.busy {
            return;
        }
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.busy = true;
        self.last = None;
        thread::spawn(move || {
            let _ = tx.send(run_install());
        });
    }
}

fn make_updater() -> Result<AxoUpdater, String> {
    let mut u = AxoUpdater::new_for(APP_NAME);
    u.set_release_source(ReleaseSource {
        release_type: ReleaseSourceType::GitHub,
        owner: GITHUB_OWNER.to_string(),
        name: GITHUB_REPO.to_string(),
        app_name: APP_NAME.to_string(),
    });
    let version: Version = UpdateUi::current_version()
        .parse()
        .map_err(|e| format!("bad current version: {e}"))?;
    u.set_current_version(version).map_err(|e| e.to_string())?;

    // Install dir = directory of the running executable. On macOS this is
    // BSnoBS.app/Contents/MacOS/, which preserves the bundle wrapper across
    // updates (only the inner binary is swapped).
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let dir = exe
        .parent()
        .ok_or_else(|| "no parent dir for current exe".to_string())?;
    u.set_install_dir(dir.to_string_lossy().into_owned());
    u.disable_installer_output();
    Ok(u)
}

fn run_check() -> UpdateMessage {
    let mut u = match make_updater() {
        Ok(u) => u,
        Err(e) => return UpdateMessage::Error(e),
    };
    match u.is_update_needed_sync() {
        Ok(false) => UpdateMessage::UpToDate,
        Ok(true) => UpdateMessage::Available {
            latest: "newer".to_string(),
        },
        Err(AxoupdateError::NoStableReleases { .. }) | Err(AxoupdateError::ReleaseNotFound { .. }) => {
            UpdateMessage::UpToDate
        }
        Err(e) => UpdateMessage::Error(e.to_string()),
    }
}

fn run_install() -> UpdateMessage {
    let mut u = match make_updater() {
        Ok(u) => u,
        Err(e) => return UpdateMessage::Error(e),
    };
    match u.run_sync() {
        Ok(Some(result)) => UpdateMessage::Installed {
            new_version: result.new_version.to_string(),
        },
        Ok(None) => UpdateMessage::NoUpdateApplied,
        Err(e) => UpdateMessage::Error(e.to_string()),
    }
}
