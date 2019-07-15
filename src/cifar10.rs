use linalg::Matrix;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader,BufRead,ErrorKind,Result};
use std::time::Instant;
use std::slice::Chunks;

pub fn training_data() -> (Matrix<u8>, Matrix<u8>) {
    read_files_matrix_4(train_files(), 50000)
}

pub fn test_data() -> (Matrix<u8>, Matrix<u8>) {
    read_files_matrix_4(test_file(), 10000)
}

pub fn read_train_data() -> (Vec<u8>, Vec<Vec<u8>>) {
    read_files_unzipped(train_files())
}

pub fn read_test_data() -> (Vec<u8>, Vec<Vec<u8>>) {
    read_files_unzipped(test_file())
} 

pub fn read_labels_data() -> Vec<String> {
    read_labels(label_file())
}

pub fn read_all() -> (Vec<(u8, Vec<u8>)>, Vec<(u8, Vec<u8>)>, Vec<String>) {
    (read_files(train_files()), read_files(test_file()), read_labels(label_file()))
}

fn read_files(files: Vec<&str>) -> Vec<(u8, Vec<u8>)> {
    files.iter().map(|f| read_file(f)).flatten().collect()
}

fn read_files_unzipped(files: Vec<&str>) -> (Vec<u8>, Vec<Vec<u8>>) {
    files.iter().map(|f| read_file(f)).flatten().unzip()
}

fn read_files_matrix(files: Vec<&str>) -> (Matrix<u8>, Matrix<u8>) {
    let now = Instant::now();

    let mut buffer_a = Vec::with_capacity(200730000);
    let mut buffer_b = Vec::with_capacity(200730000);

    let (labels, data) = files.iter()
        .map(|f| read_chunks(f))
        .map(|item| (item[0], item[1..].to_vec()))
        .fold((buffer_a, buffer_b), | (mut acc_l, mut acc_d), (label, mut data_i)| (push(&mut acc_l, label).to_vec(), extend(&mut acc_d, &mut data_i).to_vec()));

    println!("Time to read files {}", now.elapsed().as_millis());
    
    let label_matrix = Matrix{rows: labels.len(), cols: 1, data: labels};
    let data_matrix = Matrix{rows: data.len(), cols: 3072, data: data};

    (label_matrix, data_matrix)
}

fn read_files_matrix_1(files: Vec<&str>) -> (Matrix<u8>, Matrix<u8>) {
    let now = Instant::now();
    let (labels, data): (Vec<u8>, Vec<Vec<u8>>) = files.iter().flat_map(|f| read_file(f)).unzip();
    println!("Step() {}", now.elapsed().as_millis());
    let label_matrix = Matrix{rows: labels.len(), cols: 1, data: labels};
    println!("Step() {}", now.elapsed().as_millis());
    let data_matrix = Matrix{rows: data.len(), cols: 3072, data: data.into_iter().flatten().collect()};
    println!("Step() {}", now.elapsed().as_millis());
    (label_matrix, data_matrix)
}

fn read_files_matrix_2(files: Vec<&str>) -> (Matrix<u8>, Matrix<u8>) {
    let labels: Vec<u8> = files.iter().flat_map(|f| read_file_labels(f)).collect();
    let data: Vec<u8> = files.iter().flat_map(|f| read_file_data_1(f)).collect();
    let label_matrix = Matrix{rows: labels.len(), cols: 1, data: labels};
    let data_matrix = Matrix{rows: data.len(), cols: 3072, data: data};
    (label_matrix, data_matrix)
}

fn read_files_matrix_3(files: Vec<&str>) -> (Matrix<u8>, Matrix<u8>) {
    let now = Instant::now();
    let labels: Vec<u8> = files.iter().flat_map(|f| read_file_labels(f)).collect();
    let mut buffer = Vec::with_capacity(30730000);
    files.iter().for_each(|f| read_file_data_2(f, &mut buffer));
    println!("Step() {}", now.elapsed().as_millis());
    let data: Vec<u8> = buffer.chunks(3073)
        .map(|c| c[1..].to_vec())
        .flatten()
        .collect();
    println!("Step() {}", now.elapsed().as_millis());
    let label_matrix = Matrix{rows: labels.len(), cols: 1, data: labels};
    let data_matrix = Matrix{rows: data.len(), cols: 3072, data: data};
    (label_matrix, data_matrix)
}

fn read_files_matrix_4(files: Vec<&str>, capacity: usize) -> (Matrix<u8>, Matrix<u8>) {
    let mut labels_buffer = Vec::with_capacity(capacity);
    let mut data_buffer = Vec::with_capacity(capacity * 3072);

    unsafe {
        labels_buffer.set_len(capacity + 1);
        data_buffer.set_len(capacity * 3072);
    }

    files.iter().enumerate().for_each(|(i, f)| read_cifar(&mut File::open(f).unwrap(), &mut labels_buffer, &mut data_buffer, i));

    let label_matrix = Matrix{rows: labels_buffer.len(), cols: 1, data: labels_buffer};
    let data_matrix = Matrix{rows: data_buffer.len() / 3072, cols: 3072, data: data_buffer};
    (label_matrix, data_matrix)
}

fn read_file_data_2(file: &str, buffer: &mut Vec<u8>) {
    let mut file = File::open(file).expect("no file");
    file.read_to_end(buffer).expect("error reading");
}

fn read_file_data(file: &str) -> Vec<u8> {
    let now = Instant::now();
    let mut buffer = Vec::with_capacity(30730000);
    let mut file = File::open(file).expect("no file");
    file.read_to_end(&mut buffer).expect("error reading");
    let mut i: usize = 0;
    buffer.retain(|_| (i%3072 != 0, i += 3072).0);
    buffer
}

fn read_file_data_1(file: &str) -> Vec<u8> {
    let now = Instant::now();
    let mut buffer = Vec::with_capacity(30730000);
    let mut file = File::open(file).expect("no file");
    file.read_to_end(&mut buffer).expect("error reading");
    let now = Instant::now();
    let result: Vec<u8> = buffer.chunks(3073)
        .flat_map(|c| c[1..].to_vec())
        .collect();
    println!("Step() {}", now.elapsed().as_millis());
    result
}

fn read_file_labels(file: &str) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(30730000);
    let mut file = File::open(file).expect("no file");
    file.read_to_end(&mut buffer).expect("error reading");
    buffer.chunks(3073)
        .map(|c| c[0])
        .collect()
}

fn read_file(file: &str) -> Vec<(u8, Vec<u8>)> {
    let mut buffer = Vec::with_capacity(30730000);
    let mut file = File::open(file).expect("no file");
    file.read_to_end(&mut buffer).expect("error reading");

    buffer.chunks(3073)
        .map(|c| (c[0], c[1..].to_vec()))
        .collect()
}

fn read_chunks(file: &str) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(30730000);
    let mut file = File::open(file).expect("no file");
    file.read_to_end(&mut buffer).expect("error reading");

    buffer.chunks(3073).flat_map(|c| c[..].to_vec()).collect()
}

fn read_labels(file: &str) -> Vec<String> {
    let mut contents = String::new();
    let mut file = File::open(file).unwrap();
    file.read_to_string(&mut contents).expect("error reading file");
    contents.lines().map(ToOwned::to_owned).collect()
}

fn train_files() -> Vec<&'static str> {
    vec![
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_1.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_2.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_3.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_4.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_5.bin",
    ]
}

fn test_file() -> Vec<&'static str> {
    vec![
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/test_batch.bin",
    ]
}

fn label_file() -> &'static str {
    "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/batches.meta.txt"
}

fn push<'a, T>(a: &'a mut Vec<T>, b: T) -> &'a Vec<T> {
    a.push(b);
    a
}

fn extend<'a, T>(a: &'a mut Vec<T>, b: &mut Vec<T>) -> &'a Vec<T> {
    a.append(b);
    a
}

fn read_cifar<R: Read + ?Sized>(r: &mut R, buf_labels: &mut Vec<u8>, buf_data: &mut Vec<u8>, file_index: usize) {
    let mut pos_label = file_index * 10000;
    let mut pos_data = file_index * 10000 * 3072;

    loop {
        match r.read(&mut buf_labels[pos_label..(pos_label+1)]) {
            Ok(0) => { break; }
            Ok(n) => pos_label += n,
            Err(ref e) => { println!("Error"); }
            Err(e) => { println!("Error"); break; }
        }
        match r.read(&mut buf_data[pos_data..(pos_data+3072)]) {
            Ok(0) => { break; }
            Ok(n) => pos_data += n,
            Err(ref e) => { println!("Error"); }
            Err(e) => { println!("Error"); break; }
        }
    }
}
