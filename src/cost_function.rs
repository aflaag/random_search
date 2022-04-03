#[derive(Debug, Clone, Copy)]
pub enum CostFunction {
    L1,
    L2,
}

impl CostFunction {
    fn l1(o: f32, y: f32) -> f32 {
        (o - y).abs()
    }

    fn l2(o: f32, y: f32) -> f32 {
        (o - y) * (o - y)
    }

    pub fn function(&self) -> fn(f32, f32) -> f32 {
        match *self {
            Self::L1 => Self::l1,
            Self::L2 => Self::l2,
        }
    }
}