use godot::prelude::*;
use pyo3::prelude::*;

struct AiBridgeExtension;

#[gdextension]
unsafe impl ExtensionLibrary for AiBridgeExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
pub struct AiBridge {
    base: Base<Node>,
}

#[godot_api]
impl INode for AiBridge {
    fn init(base: Base<Node>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl AiBridge {
    #[func]
    pub fn predict_action(&self, state_json: GString) -> Variant {
        let state_str = state_json.to_string();

        let result = Python::with_gil(|py| -> PyResult<String> {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            
            // Add paths where the python module might be located
            path.call_method1("append", ("./R-NaD",))?;
            path.call_method1("append", ("/home/ubuntu/src/R-NaD-StS2/R-NaD",))?;
            
            let my_ai_module = py.import("rnad_bridge")?;

            let predict_fn = my_ai_module.getattr("predict_action")?;
            let args = (state_str,);
            let action_json: String = predict_fn.call1(args)?.extract()?;
            eprintln!("[AiBridge-Rust] predict_action returning to Godot: {}", action_json);
            
            Ok(action_json)
        });

        match result {
            Ok(json) => json.to_variant(),
            Err(e) => {
                eprintln!("[AiBridge-Rust] Python Error: {:?}", e);
                "{\"error\": \"python_fault\"}".to_variant()
            }
        }
    }
    #[func]
    pub fn check_screenshot_request(&self) -> Variant {
        let result = Python::with_gil(|py| -> PyResult<String> {
            let my_ai_module = py.import("rnad_bridge")?;
            let check_fn = my_ai_module.getattr("check_screenshot_request")?;
            let path: String = check_fn.call0()?.extract()?;
            Ok(path)
        });

        match result {
            Ok(path) => path.to_variant(),
            Err(e) => {
                eprintln!("[AiBridge-Rust] Python Error in check_screenshot_request: {:?}", e);
                "".to_variant()
            }
        }
    }

    #[func]
    pub fn mark_screenshot_done(&self) -> Variant {
        let result = Python::with_gil(|py| -> PyResult<bool> {
            let my_ai_module = py.import("rnad_bridge")?;
            let mark_fn = my_ai_module.getattr("mark_screenshot_done")?;
            let res: bool = mark_fn.call0()?.extract()?;
            Ok(res)
        });

        match result {
            Ok(res) => res.to_variant(),
            Err(e) => {
                eprintln!("[AiBridge-Rust] Python Error in mark_screenshot_done: {:?}", e);
                false.to_variant()
            }
        }
    }
}

