pub enum ActivationFunction {
    Sigmoid,
    ReLU,
}

impl ActivationFunction {
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + x.exp())
    }

    fn relu(x: f32) -> f32 {
        if x >= 0.0 {
            x
        } else {
            0.0
        }
    }

    pub fn value(&self) -> fn(f32) -> f32 {
        match *self {
            Self::Sigmoid => Self::sigmoid,
            Self::ReLU => Self::relu,
        }
    }
}