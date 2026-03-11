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
            let my_ai_module = PyModule::from_code(
                py,
                r#"
def predict(state_json):
    print(f"Python received state: {state_json}")
    return '{"action": "test"}'
"#,
                "ai_logic.py",
                "ai_logic",
            )?;

            let predict_fn = my_ai_module.getattr("predict")?;
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
