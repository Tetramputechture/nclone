fn main() {
    use nclone_native::entities::Entity;
    println!("size_of Entity = {}", std::mem::size_of::<Entity>());
    println!("size_of Segment = {}", std::mem::size_of::<nclone_native::segments::Segment>());
    println!("size_of Simulator = {}", std::mem::size_of::<nclone_native::simulator::Simulator>());
}
