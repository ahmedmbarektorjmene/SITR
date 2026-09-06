use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::app_state::AppState;

pub struct Scheduler {
    state: Arc<Mutex<AppState>>,
    running: Arc<Mutex<bool>>,
}

impl Scheduler {
    pub fn new(state: Arc<Mutex<AppState>>) -> Self {
        Self {
            state,
            running: Arc::new(Mutex::new(false)),
        }
    }

    pub fn start(&self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = true;
        }

        let state = Arc::clone(&self.state);
        let running = Arc::clone(&self.running);

        std::thread::Builder::new()
            .name("porda-scheduler".to_string())
            .spawn(move || {
                tracing::info!("Scheduler thread started");

                loop {
                    if !*running.lock().unwrap() {
                        break;
                    }

                    let interval = {
                        let s = state.lock().unwrap();
                        if s.is_active {
                            Duration::from_millis(s.detection_interval_ms())
                        } else {
                            Duration::from_millis(1000)
                        }
                    };

                    std::thread::sleep(interval);
                }

                tracing::info!("Scheduler thread stopped");
            })
            .expect("Failed to spawn scheduler thread");
    }

    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }
}
