#!/usr/bin/env ocaml
#use "topfind";;
#require "yojson";;


let zefile = try Sys.argv.(1) with _ -> (
   Printf.fprintf stderr "usage: %s <file.json>\n" Sys.argv.(0);
   exit 1
);;

let gg = try Yojson.Basic.from_file zefile with _ -> (
   Printf.fprintf stderr "Error while loading %s\n" zefile;
   exit 1
) ;;

let string_of_assoc_key (assoc:Yojson.Basic.t) key = try (
   match assoc with
   | `Assoc al -> (
      match List.assoc key al with
      | `String s -> s
      | _ -> assert false
   )
   | _ -> assert false
) with _ -> (
   Printf.fprintf stderr "Error: no (string) value associated to key '%s'\n" key;
   exit 1
)
let list_of_assoc_key (assoc:Yojson.Basic.t) key = try (
   match assoc with
   | `Assoc al -> (
      match List.assoc key al with
      | `List l -> l
      | _ -> assert false
   )
   | _ -> assert false
) with _ -> (
   Printf.fprintf stderr "Error: no (list) value associated to key '%s'\n" key;
   exit 1
)

module H = Hashtbl
module P = Printf

let nodes2tab nodes = (
   let nodetab = H.create 200 in
   let parse_node (i:int) (n: Yojson.Basic.t) = (
      let nme = string_of_assoc_key n "name" in
      let raw_inputs = list_of_assoc_key n "inputs" in
      let getinindex : Yojson.Basic.t -> int  =
         function `List ((`Int i)::_) -> i
         | _ -> assert false
      in
      let inputs = List.map getinindex raw_inputs in
      (* H.add nodetab i raw_inputs; *)
      H.add nodetab i (nme,inputs);

      (* Printf.fprintf stderr "node %d = %s ; inputs: " i nme; *)
      (* List.iter (fun i -> Printf.fprintf stderr "%d " i) inputs; *)
      (* List.iter (fun i -> Printf.fprintf stderr "%s " (Yojson.Basic.to_string i)) raw_inputs; *)
      (* Printf.fprintf stderr "\n" *)
   ) in
   List.iteri parse_node nodes ;
   nodetab
)

let tab2dot tab = (
   P.printf "digraph zegraph {\n";
   let max = (H.length tab) -1 in
   for x = 0 to max do
      let (nme,ins) = H.find tab x in
      (* a [label="Foo"]; *)
      P.printf "%d [label=\"%s\"];\n" x nme;
      List.iter (fun p -> P.printf "%d -> %d;\n" p x) ins
   done;
   P.printf "}\n"
)

let nodes = list_of_assoc_key gg "nodes" ;;

let tab = nodes2tab nodes ;;

let _ = tab2dot tab;;
