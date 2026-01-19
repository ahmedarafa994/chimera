/**
 * Quicksort implementation in JavaScript
 * @param {number[]} arr - Array to sort
 * @returns {number[]} - Sorted array
 */
function quicksort(arr) {
  if (arr.length <= 1) {
    return arr;
  }

  const pivot = arr[Math.floor(arr.length / 2)];
  const left = [];
  const middle = [];
  const right = [];

  for (const element of arr) {
    if (element < pivot) {
      left.push(element);
    } else if (element > pivot) {
      right.push(element);
    } else {
      middle.push(element);
    }
  }

  return [...quicksort(left), ...middle, ...quicksort(right)];
}

module.exports = { quicksort };
