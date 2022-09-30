use pyo3::prelude::*;
extern crate wake;
use polars::prelude::*;
use polars::prelude::DataFrame;
use std::env;
use wake::graph::*;

pub mod tpch;
pub mod utils;

use tpch::*;

#[pyfunction]
fn test_function() -> String {
    "Hello World from Rust".to_string()
}

#[pyfunction]
fn main() {
    // Arguments:
    // 0: Whether to run query or test. Required. query/test.
    // 1: Query Number. Required.
    // 2: Scale of the TPC-H Dataset. Optional. Default: 1.
    // 3: Directory containing the dataset. Optional. Default: resources/tpc-h/data/scale=1/partition=1/

    env_logger::Builder::from_default_env()
        .format_timestamp_micros()
        .init();

    let args = env::args().skip(1).collect::<Vec<String>>();
    let mut my_vec: Vec<String> = Vec::new();
    let mut my_vec1: Vec<String> = Vec::new();
    let name1 = String::from("query");
    let name2 = String::from("q1");
    my_vec.push(name1);
    my_vec1.push(name2);
    run_query(my_vec1);
}

fn run_query(args: Vec<String>) {
    if args.len() == 0 {
        panic!("Query not specified. Run like: cargo run --release --example tpch_polars -- q1")
    }
    let query_no = args[0].as_str();
    let scale = if args.len() <= 1 {
        1
    } else {
        *(&args[1].parse::<usize>().unwrap())
    };
    let data_directory = if args.len() <= 2 {
        "../../resources/tpc-h/data/scale=1/partition=1"
    } else {
        args[2].as_str()
    };
    let mut output_reader = NodeReader::empty();
    let mut query_service = get_query_service(query_no, scale, data_directory, &mut output_reader);
    log::info!("Running Query: {}", query_no);
    utils::run_query(&mut query_service, &mut output_reader);
}

pub fn get_query_service(
    query_no: &str,
    scale: usize,
    data_directory: &str,
    output_reader: &mut NodeReader<DataFrame>,
) -> ExecutionService<DataFrame> {
    let table_input = utils::load_tables(data_directory, scale);
    // TODO: UNCOMMENT THE MATCH STATEMENTS BELOW AS YOU IMPLEMENT THESE QUERIES.
    let query_service = match query_no {
        "q1" => q1::query(table_input, output_reader),
        "q14" => q14::query(table_input, output_reader),
        _ => panic!("Invalid Query Parameter"),
    };
    query_service
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn tpch_polars(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(main, m)?)?;
    Ok(())
}