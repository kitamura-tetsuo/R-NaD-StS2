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
    pub fn predict_action(&self, state_json: GString) -> GString {
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
            
            Ok(action_json)
        });

        match result {
            Ok(json) => GString::from(&json),
            Err(e) => {
                godot_error!("Python Error: {:?}", e);
                GString::from("{\"error\": \"python_fault\"}")
            }
        }
    }
}
