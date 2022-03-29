#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
}

impl ActivationFunction {
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn relu(x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            0.0
        }
    }

    pub fn function(&self) -> fn(f64) -> f64 {
        match *self {
            Self::Sigmoid => Self::sigmoid,
            Self::ReLU => Self::relu,
        }
    }
}