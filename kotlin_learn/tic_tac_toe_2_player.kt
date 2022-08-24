package tictactoe

fun printGrid(istr: String) : Unit {
    println("---------")
    println("| ${istr[0]} ${istr[1]} ${istr[2]} |")
    println("| ${istr[3]} ${istr[4]} ${istr[5]} |")
    println("| ${istr[6]} ${istr[7]} ${istr[8]} |")
    println("---------")
}

fun checkState(istr: String) : String {
    // var to store result string
    var res: String = "Impossible"
    // vars to store number of x, o, empty
    var nx: Int = 0
    var no: Int = 0
    var ne: Int = 0
    // count number of x, number of o and number of empty spaces
    for (ch in istr) {
        if (ch == 'X') {
            nx += 1
        } else if (ch == 'O') {
            no += 1
        } else {
            ne += 1
        }
    }
    // check for impossibility
    if (nx - no > 1 || no - nx > 1) {
        res = "Impossible"
        // println(res)
        return res
    }
    // check if there are any 3 consecutive xs
    // check if there are any 3 consecutive os
    var xcons: Boolean = false
    var ocons: Boolean = false
    var n3: Int = 0
    // check in same column
    for (i in 0..2) {
        var prev = istr[i]
        var cur: Char = ' '
        var f = true
        for (j in i..8 step 3) {
            cur = istr[j]
            if (cur != prev) {
                f = false
                break
            }
            prev = cur
        }
        if (f) {
            if (cur == 'X') {
                xcons = true
                n3 += 1
            } else if (cur == 'O') {
                ocons = true
                n3 += 1
            }
        }
    }
    // check in same row
    for (i in 0..8 step 3) {
        var prev = istr[i]
        var cur: Char = ' '
        var f = true
        for (j in i..i+2) {
            cur = istr[j]
            if (cur != prev) {
                f = false
                break
            }
            prev = cur
        }
        if (f) {
            if (cur == 'X') {
                xcons = true
                n3 += 1
            } else if (cur == 'O') {
                ocons = true
                n3 += 1
            }
        }
    }
    // check leading diagonal
    var prev = istr[0]
    var cur: Char = ' '
    var ffd = true
    for (i in 0..8 step 4) {
        cur = istr[i]
        if (cur != prev) {
            ffd = false
            break
        }
        prev = cur
    }
    if (ffd) {
        if (cur == 'X') {
            xcons = true
            n3 += 1
        } else if (cur == 'O') {
            ocons = true
            n3 += 1
        }
    }
    // check opposite diagonal
    prev = istr[2]
    cur  = ' '
    var fbd = true
    for (i in 2..6 step 2) {
        cur = istr[i]
        if (cur != prev) {
            fbd = false
            break
        }
        prev = cur
    }
    if (fbd) {
        if (cur == 'X') {
            xcons = true
            n3 += 1
        } else if (cur == 'O') {
            ocons = true
            n3 += 1
        }
    }
    if (!xcons && !ocons && ne > 0) {
        res = "Game not finished"
    } else if (!xcons && !ocons && ne == 0) {
        res = "Draw"
    } else if (xcons && !ocons && n3 == 1) {
        res = "X wins"
    } else if (!xcons && ocons && n3 == 1) {
        res = "O wins"
    } else if (xcons && ocons || n3 >  1) {
        res = "Impossible"
    }
    return res
}

fun receiveValidInput(istr: String, curPlayer: Char) : String {
    var isValidInput: Boolean = false
    var nstr: String = istr
    do {
        val li = readln().split(" ").map{ it.toIntOrNull() }.toMutableList()
        if (li.size != 2) {
            println("Coordinates should be in the 2D plane!")
            continue
        }
        if (li[0] == null || li[1] == null) {
            println("You should enter numbers!")
            continue
        }
        val x: Int = li[0]!!
        val y: Int = li[1]!!
        if (x < 1 || x > 3 || y < 1 || y > 3) {
            println("Coordinates should be from 1 to 3!")
            continue
        }
        val fidx: Int = (x - 1)*3 + (y - 1)
        if (istr[fidx] != ' ') {
            println("This cell is occupied! Choose another one!")
            continue
        }
        nstr = istr.substring(0, fidx) + "$curPlayer" + istr.substring(fidx + 1)
        isValidInput = true
    } while (!isValidInput);
    return nstr
}

fun main() {
    // write your code here
    val startState = "         "
    // print the start state
    printGrid(startState)
    // check initial state for game end
    var res: String = checkState(startState)
    var curState: String = startState
    var nextState: String = " "
    var curPlayer: Char = 'X'
    while (res == "Game not finished") {
        // receive input
        nextState = receiveValidInput(curState, curPlayer)
        // print grid after new move
        printGrid(nextState)
        // check for any wins/draws or impossible states
        res = checkState(nextState)
        // update vars for next loop
        curState = nextState
        if (curPlayer == 'X') {
            curPlayer = 'O'
        } else {
            curPlayer = 'X'
        }
    }
    println(res)
}
