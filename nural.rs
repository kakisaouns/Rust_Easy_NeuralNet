use std::ops::Fn;

// general
fn relu(x : &Vec<f64>) -> Vec<f64>{
    let mut ret = x.to_vec();
    for i in &mut ret{
        *i = if 0.0 < *i {*i} else {0.0};
    }
    ret
}

fn num_diff<F>(f:&F,w:&mut Vec<Vec<Vec<f64>>>,tgt_p1:usize,tgt_p2:usize) -> Vec<f64> where F : (Fn(&Vec<Vec<Vec<f64>>>) -> f64){
    let h = 1e-04;
    let mut grad = vec![0.0_f64;w[tgt_p1][tgt_p2].len()];
    
    for i in 0..w[tgt_p1][tgt_p2].len(){
        let tmp = w[tgt_p1][tgt_p2][i];

        w[tgt_p1][tgt_p2][i] = tmp + h;
        let fw = f(&w);

        w[tgt_p1][tgt_p2][i] = tmp - h;
        let bk = f(&w);

        grad[i] = (fw - bk) / (2.0 * h);
        w[tgt_p1][tgt_p2][i] = tmp;
    }

    grad
}

fn mean_squared_error(y:&Vec<f64>,t:&Vec<f64>) -> f64{
    if y.len() != t.len(){
        panic!("unmatch dim. y:{},t:{}",y.len(),t.len());
    }
    
    let mut ret= 0.0_f64;
    for i in 0..y.len(){
        ret += (y[i] - t[i]).powf(2.0);
    }

    let ret_val = ret / 2.0;

    return ret_val;
}

fn dot(x:&Vec<f64>,w:&Vec<Vec<f64>>) -> Vec<f64>{
    if w.len() <= 0 || x.len() != w.len(){
        panic!("unmatch dim. x:{},y:{}",x.len(),w.len());
    }

    let mut ret = vec![0.0_f64;w[0].len()];

    for i in 0..w[0].len(){
        for j in 0..x.len(){
            ret[i] += x[j] * w[j][i];
        }
    }

    ret
}

fn predict(x:&Vec<f64>,w:&Vec<Vec<Vec<f64>>>) -> Vec<f64>{
    let x1 = dot(x,&w[0]);
    let z1 = relu(&x1);
    let x2 = dot(&z1,&w[1]);
    let z2 = relu(&x2);
    let x3 = dot(&z2,&w[2]);

    x3
}

fn backward(x:&Vec<f64>,t:&Vec<f64>,w:&mut Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
    let lt = 0.05;
    let f = |wh:&Vec<Vec<Vec<f64>>>| -> f64{
        let pr = predict(x,wh);
        let ret = mean_squared_error(&pr,t);
        return ret;
    };

    let mut diff_ret : Vec<Vec<Vec<f64>>> = Vec::new();

    for i in 0..w.len(){
        diff_ret.push(vec![Vec::new();w[i].len()]);
        for j in 0..w[i].len(){
            diff_ret[i][j] = num_diff(&f,w,i,j);
            for k in 0..diff_ret[i][j].len(){
                w[i][j][k] -= lt * diff_ret[i][j][k];
            }
        }
    }
    

    diff_ret
}

fn forward(x:&Vec<f64>,w:&Vec<Vec<Vec<f64>>>) -> Vec<f64>{
    predict(x,w)
}

fn training(input : &Vec<Vec<f64>>,answer : &Vec<Vec<f64>>,weight : &mut Vec<Vec<Vec<f64>>>,epoch :usize,view_status :usize){
    let epoch_val = epoch + 1;
    let view_status_val = if view_status <= 0 {1} else {view_status};
    for i in 1..epoch_val{
        for dt in 0..input.len(){
            backward(&input[dt],&answer[dt],weight);
        }
        if i % view_status_val == 0{
            println!("epoch:{},end",i);
        }
    }
}

fn test(input : &Vec<Vec<f64>>,weight : &Vec<Vec<Vec<f64>>>){
    for inp in input{
        let ret = forward(&inp,&weight);
        for i in ret{
            println!("ret[{},{}]:{}",inp[0],inp[1],i);
        }
    }
}

fn main(){
    let input = vec![
        vec![0.0_f64,0.0_f64],
        vec![1.0_f64,0.0_f64],
        vec![0.0_f64,1.0_f64],
        vec![1.0_f64,1.0_f64]
    ];
    let mut weight = vec![
        vec![vec![0.1_f64,0.10_f64],vec![0.1_f64,0.1_f64]],
        vec![vec![0.1_f64,0.14_f64],vec![0.1_f64,0.1_f64]],
        vec![vec![0.1_f64;1],vec![0.1_f64;1]]];
    let res = vec![
        vec![0.0_f64],
        vec![1.0_f64],
        vec![1.0_f64],
        vec![0.0_f64]
    ];

    training(&input,&res,&mut weight,30000,100);
    test(&input,&weight);
}
